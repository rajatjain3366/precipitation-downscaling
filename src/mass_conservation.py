"""
Global precipitation mass-conservation regularization for QRE ("QRE with global
mass-conservation regularization"). Only the domain total of daily rainfall is
constrained; no moisture-flux, PDE or other physics.

For day t, valid mask M, cell area A_i, observed P and predicted P_hat (mm/day):

    O_t = sum_i M_i A_i P_i,t          S_t = sum_i M_i A_i P_hat_i,t
    relative: L_mass     = 1/B sum_t |S_t - O_t| / (O_t + eps')
    absolute: L_mass_abs = 1/B sum_t |S_t - O_t| / sum_i M_i A_i     (area-mean mm/day)

Implemented with normalized weights w_i = M_i A_i / sum_j M_j A_j, i.e. on
area-mean totals o_t = O_t / sum M A and s_t = S_t / sum M A. The relative form
is then |s_t - o_t| / (o_t + eps) with eps in mm/day (eps' = eps * sum M A),
identical to the formula above but numerically safer in float32.

Targets are physical mm/day (the pipeline never normalizes them), so there is
nothing to undo. Masked cells are removed with tf.where before any arithmetic,
so NaN or any value there cannot leak into the loss or its gradient.
"""
import numpy as np
import tensorflow as tf

from src.ConvolutionalNetworks import BG_mean, unpack_bgout
from src.GammaLoss import GammaLoss
from src.quantiles import OmegaLoss


class MassConservation:
    def __init__(self, areas, mask=None, mode="relative", eps=0.1):
        if mode not in ("relative", "absolute"):
            raise ValueError(f"mass mode must be 'relative' or 'absolute', got {mode!r}")
        areas = np.asarray(areas, dtype=np.float64)
        if np.any(areas <= 0):
            raise ValueError("cell areas must be positive")
        mask = np.ones(areas.shape, bool) if mask is None else np.asarray(mask, bool)
        w = np.where(mask, areas, 0.0)
        self.total_area_km2 = float(w.sum())
        self.weights = tf.constant((w / w.sum()).astype(np.float32))  # (D,)
        self.mask = tf.constant(mask)
        self.mode, self.eps = mode, float(eps)

    def area_mean(self, y):
        """Area-weighted domain mean (mm/day) of each day, (B,)."""
        y = tf.where(self.mask, tf.cast(y, tf.float32), 0.0)
        return tf.reduce_sum(y * self.weights, axis=-1)

    def per_day(self, y_true, y_pred, mode=None):
        """|S_t - O_t| term of each day, (B,); never summed across days."""
        o, s = self.area_mean(y_true), self.area_mean(y_pred)
        if (mode or self.mode) == "relative":
            return tf.abs(s - o) / (o + self.eps)
        return tf.abs(s - o)

    def __call__(self, y_true, y_pred, mode=None):
        return tf.reduce_mean(self.per_day(y_true, y_pred, mode))


def bg_expected(params):
    """p * alpha * beta from BGNet output (B, D, 3) -> (B, D), mm/day."""
    p, a, b = unpack_bgout(params)
    return tf.squeeze(BG_mean(p, a, b, mode="expected"), axis=-1)


class MemberLossWithMass(tf.keras.losses.Loss):
    """
    Member loss: Bernoulli-Gamma NLL + lambda * L_mass on the member's p*alpha*beta.
    With lambda = 0 it returns the NLL alone (same op graph as GammaLoss).
    """

    def __init__(self, nll, mass, lam):
        super().__init__()
        self.nll, self.mass, self.lam = nll, mass, float(lam)

    def call(self, y_true, y_pred):
        loss = self.nll(y_true, y_pred)
        if self.lam != 0.0:
            loss = loss + self.lam * self.mass(y_true, bg_expected(y_pred))
        return loss


def member_metrics(nll, mass):
    """Keras metrics that log the two terms of MemberLossWithMass separately."""
    def bg_nll(y_true, y_pred):
        return nll(y_true, y_pred)

    def mass_loss(y_true, y_pred):
        return mass(y_true, bg_expected(y_pred))

    return [bg_nll, mass_loss]


class JointEnsemble(tf.keras.Model):
    """
    Joint fine-tuning of the ensemble g(x) = sum_i w_i(x) f_i(x):
        loss = L_QRE + lambda * L_mass(y, g)
    L_QRE is the Weight Network's loss (squared EMD by default) on the interval
    labels; if the members are trained too, it also contains each member's
    Bernoulli-Gamma NLL on the batch days of its own interval (the members' own
    stage loss). The two terms are logged as qre_loss and mass_loss.

    Inputs: [x, F] with F (B, N, D) the precomputed member predictions (members
    frozen), or [x] with `members` given (frozen or trainable). Targets: [labels, y].
    """

    def __init__(self, omega, wn_loss, mass, lam, members=None, train_members=False, nll=None):
        super().__init__()
        self.omega, self.wn_loss, self.mass, self.lam = omega, wn_loss, mass, float(lam)
        self.members = members or []
        self.train_members = bool(train_members and members)
        self.nll = nll
        if self.train_members and nll is None:
            raise ValueError("training the members jointly needs their NLL loss")
        for m in self.members:
            m.trainable = self.train_members

    def call(self, inputs, training=False):
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        w = self.omega(x, training=training)
        if self.members:
            params = [m(x, training=training and self.train_members) for m in self.members]
            f = tf.stack([bg_expected(p) for p in params], axis=1)
        else:
            params, f = None, tf.cast(inputs[1], tf.float32)
        return w, f, params

    def loss_terms(self, inputs, labels, y, training):
        """(total, L_QRE, L_mass) of a batch. (Not `_losses`: Keras uses that name internally.)"""
        w, f, params = self(inputs, training=training)
        g = tf.einsum("bn,bnd->bd", w, f)
        qre = tf.reduce_mean(self.wn_loss(labels, w))
        if self.train_members:
            for i, p in enumerate(params):
                pa, a, b = (tf.squeeze(t, -1) for t in unpack_bgout(p))
                nll_day = -tf.reduce_mean(GammaLoss.lBG_distribution(
                    y, pa, tf.exp(a), tf.exp(b), self.nll.y_thrs, self.nll.epsilon), axis=-1)
                own = labels[:, i]
                qre = qre + tf.reduce_sum(own * nll_day) / tf.maximum(tf.reduce_sum(own), 1.0)
        mass = self.mass(y, g)
        total = qre + self.lam * mass if self.lam != 0.0 else qre
        return total, qre, mass

    def train_step(self, data):
        inputs, (labels, y) = data[0], data[1]
        with tf.GradientTape() as tape:
            total, qre, mass = self.loss_terms(inputs, labels, y, training=True)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total, "qre_loss": qre, "mass_loss": mass}

    def test_step(self, data):
        inputs, (labels, y) = data[0], data[1]
        total, qre, mass = self.loss_terms(inputs, labels, y, training=False)
        return {"loss": total, "qre_loss": qre, "mass_loss": mass}


def weight_network_loss(wcfg, n_intervals):
    return OmegaLoss(num_classes=n_intervals, class_reduction=wcfg["emd_reduction"], kind=wcfg["loss"],
                     ce_weight=wcfg["ce_weight"], emd_weight=wcfg["emd_weight"])
