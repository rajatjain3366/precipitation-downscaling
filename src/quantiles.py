import xarray as xr
import numpy as np
import tensorflow as tf
import os

from src.useful_functions import *
from src.Modules import *
from src.segmentation import assign_intervals, daily_totals, one_hot


# discretise the dataset, taking y sample to bin index
# thresholds: daily-total thresholds t_1..t_{N-1} fitted on the TRAINING split
#   (src/segmentation.py). Pass them for every split so validation/test days are
#   labelled with the training boundaries (fixes H5 and M1).
# thresholds=None keeps the original behaviour: boundaries recomputed from y, and
#   each boundary is the SUM of all days tied at that rounded total (bug H5).
def bin_y_var(y, Q_ranges, thresholds=None):
    if thresholds is not None:
        if len(thresholds) != len(Q_ranges) - 1:
            raise ValueError(f"need {len(Q_ranges) - 1} thresholds, got {len(thresholds)}")
        return list(one_hot(assign_intervals(daily_totals(y), thresholds), len(Q_ranges)))

    bins = []

    # get bin limits
    for ql, qh in Q_ranges:
        yq = data_between(y, qh)
        bins.append(np.sum(yq))

    # get quantile ranges
    quantiles = []
    for y_sample in y:
        ys_sum = np.sum(y_sample)
        bin_index = len(bins) - 1
        for n, bin_thrs in enumerate(bins):
            if ys_sum <= bin_thrs:
                bin_index = n
                break

        new_y = np.zeros(len(Q_ranges))
        new_y[bin_index] = 1

        quantiles.append(new_y)

    return quantiles


def weight(y1, y2):
    return np.abs(y1 - y2)


# implements EMD loss for multiclass classification
class OmegaLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        num_classes,
        p=1,
        weight_func=weight,
        class_reduction="mean",  # "sum" = paper Eq. 6; "mean" (original code) divides by N
        kind="emd",              # see call(); "emd" is the paper's loss
        ce_weight=1.0,
        emd_weight=1.0,
    ):
        super(OmegaLoss, self).__init__()

        if class_reduction not in ("sum", "mean"):
            raise ValueError(f"class_reduction must be 'sum' or 'mean', got {class_reduction!r}")
        if kind not in ("emd", "squared_emd", "cross_entropy", "ce_plus_emd"):
            raise ValueError(f"unknown Weight Network loss kind {kind!r}")
        self.class_reduction = class_reduction
        self.kind = kind
        self.ce_weight = ce_weight
        self.emd_weight = emd_weight
        self.num_classes = num_classes
        self.p = p
        self.weight_func = weight_func
        self.weight_matrix = self.gen_matrix()

    def gen_matrix(self):
        matrix = np.array(
            [
                [self.weight_func(n, m) for n in range(self.num_classes)]
                for m in range(self.num_classes)
            ]
        )
        matrix = tf.constant(matrix, dtype=tf.float32)
        return tf.transpose(matrix)

    # kind="emd":             paper Eq. 6, sum_k |k - k'|^p * w_k. Linear in w, so its
    #                          minimum is one-hot and its softmax gradient w_k (c_k - L)
    #                          vanishes there (observed: Weight Network collapse).
    # kind="squared_emd":      squared EMD of Hou et al. (2016), sum_k (CDF_w(k) - CDF_y(k))^2
    # kind="cross_entropy":    -sum_k y_k log w_k
    # kind="ce_plus_emd":      ce_weight * cross-entropy + emd_weight * Eq. 6
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.kind == "squared_emd":
            diff = tf.cumsum(y_pred, axis=1) - tf.cumsum(y_true, axis=1)
            return tf.reduce_sum(tf.square(diff), axis=1)
        if self.kind == "cross_entropy":
            return self.cross_entropy(y_true, y_pred)
        if self.kind == "ce_plus_emd":
            return self.ce_weight * self.cross_entropy(y_true, y_pred) + self.emd_weight * self.emd(y_true, y_pred)
        return self.emd(y_true, y_pred)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        return -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=1)

    def emd(self, y_true, y_pred):
        col_indices = tf.argmax(y_true, axis=1)
        weight_cols = tf.pow(tf.gather(self.weight_matrix, col_indices), self.p)

        # Eq. 6: L = sum_k |k - k'|^p * omega_k
        if self.class_reduction == "sum":
            emd = tf.reduce_sum(weight_cols * y_pred, axis=1)
        else:
            emd = tf.reduce_mean(weight_cols * y_pred, axis=1)

        return emd


# The weight network omega_theta
class Omega(tf.keras.Model):
    def __init__(
        self,
        output_dim,
        kernels=[6, 6, 6],
        channels=[64, 128, 256],
        padding="valid",
        init_conv=False,     # True = same backbone as BGNet, including its 3x3 init conv
        dense_units=100,
        dropout=0.2,
    ):

        super(Omega, self).__init__()

        self.output_dim = output_dim
        self.kernels = kernels
        self.channels = channels
        self.padding = padding
        self.init_conv = init_conv
        self.dense_units = dense_units
        self.dropout = dropout

        self._omega = self.omega()

    def call(self, x, training=False):
        x = self._omega(x)

        return x

    def omega(self):
        omega = tf.keras.Sequential()

        if self.init_conv:
            omega.add(tf.keras.layers.Conv2D(self.channels[0], 3, activation="relu", padding="same"))
        for ks, nch in zip(self.kernels, self.channels):
            omega.add(tf.keras.layers.Conv2D(nch, ks, activation="relu", padding=self.padding))
            omega.add(ChannelAttentionModule(nch, 1, True, True))
            omega.add(tf.keras.layers.Conv2D(nch, ks, activation="relu", padding=self.padding))
            # omega.add(tf.keras.layers.MaxPool2D(2))

        omega.add(tf.keras.layers.Flatten())
        omega.add(tf.keras.layers.Dense(self.dense_units, activation="relu"))
        omega.add(tf.keras.layers.Dropout(self.dropout))
        omega.add(tf.keras.layers.Dense(self.dense_units, activation="relu"))
        omega.add(tf.keras.layers.Dropout(self.dropout))
        omega.add(tf.keras.layers.Dense(self.output_dim, activation="softmax"))

        return omega


# quantile boost ensemble g
class ynetwork:
    def __init__(self, CNN_dict, weight_model=None, fixed_weights=None):

        self.CNN_dict = CNN_dict

        if weight_model is not None:
            self.weights = weight_model
        elif fixed_weights is not None:
            self.weights = lambda x: fixed_weights

    def __call__(self, x):
        pred = None
        for CNN_list in self.CNN_dict.values():
            mem = evaluate_models(CNN_list, x, take_mean=True)
            # Ensure specialist prediction has shape (batch, output_dim)
            if len(mem.shape) == 3 and mem.shape[-1] == 1:
                mem = tf.squeeze(mem, axis=-1)
            mem = tf.expand_dims(mem, axis=1)
            if pred is None:
                pred = mem
            else:
                pred = tf.concat((pred, mem), axis=1)

        # weights: (batch, num_specialists, 1)
        # pred:    (batch, num_specialists, output_dim)
        weights = tf.expand_dims(self.weights(x), axis=-1)
        weights = tf.cast(weights, dtype=tf.float32)
        x = tf.reduce_sum(weights * pred, axis=1)

        return x
