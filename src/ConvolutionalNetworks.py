import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp

from src.useful_functions import * 
from src.Modules import *
from src.GammaLoss import *
from src.quantiles import * 

def unpack_bgout(y_pred):
    # unpacks network output 
    p = tf.gather(y_pred, indices=[0], axis=-1)
    alpha =  tf.gather(y_pred, indices=[1], axis=-1)
    beta = tf.gather(y_pred, indices=[2], axis=-1)  
    
    return p, alpha, beta

def BG_mean(p, alpha, beta, mode="expected"):
    # mean of BG net distribution
    # mode="expected":  p * alpha * beta, the paper's Eq. 3 (MSE-optimal)
    # mode="threshold": 1[p > 0.5] * alpha * beta, the original repo's rule
    if mode == "expected":
        rain = p
    elif mode == "threshold":
        rain = tf.where(p > 0.5, 1.0, 0.0)
    else:
        raise ValueError(f"unknown BG_mean mode {mode!r}")
    mean = rain * tf.exp(alpha) * tf.exp(beta)

    return mean


class BGNet(tf.keras.Model):
    def __init__(
                        self,
                        output_dim,                       # number of grid cells over domain 
                        coords,                           # coords from the xr.DataArray 
                        baseline=True,                    # if True, use BGNet(-)
                        avg_pool=True,                    # determines which pooling types to use  
                        max_pool=False,
                        num_dense=[256],                  # dense layers in the first dense layer 
                        kernel_sizes=[3, 3, 3],           # kernel sizes of each convolutional block 
                        num_channels=[64, 128, 256],      # channel sizes of convolutional block 
                        drop_out_rate=0.2,                # dropout rate of hidden layer 
                        reduction_rate=1,                 # reduction rate of channel attention 
                        n=1,                              # size of predictor samples in n x m 
                        m=1,
                        aux_data=None,                    # if aux module should be used
                        aux_mode='embed',
                        conv_padding='valid',             # 'same' keeps 16x16 CORDEX inputs from shrinking away
                        dense_activation='gelu'           # paper: ReLU in all hidden layers
                 ):
        
        # represents all BGNets; BGNet, BGNet(-), Embed, Interp

        super(BGNet, self).__init__()

        self.output_dim = output_dim
        self.coords = coords
        
        self.baseline = baseline
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.num_dense = num_dense
        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.drop_out_rate = drop_out_rate
        self.reduction_rate = reduction_rate
        self.dense_activation = dense_activation

        self.init_conv = tf.keras.layers.Conv2D(self.num_channels[0], 3, activation='relu', padding='same') 
        
        # convolutional part 
        self.ds_module = DownScaleModule( 
                                                self.baseline,
                                                self.avg_pool, 
                                                self.max_pool,
                                                self.num_channels, 
                                                self.kernel_sizes,
                                                self.reduction_rate,
                                                aux_mode=aux_mode,
                                                # accepts np.ndarray or xr.DataArray
                                                aux_data=None if aux_data is None else np.asarray(getattr(aux_data, 'values', aux_data), dtype=np.float32),
                                                n=n,
                                                m=m,
                                                padding=conv_padding
                                        )
            
        # first dense layer 
        self.dense_module = self.feed_forward_module()
        self.drop_out = tf.keras.layers.Dropout(self.drop_out_rate)
        
        # dense branches for the distributions parameters 
        self.p = tf.keras.layers.Dense(self.output_dim, activation='sigmoid')
        self.beta = tf.keras.layers.Dense(self.output_dim, activation=self.activation_beta)
        self.alpha = tf.keras.layers.Dense(self.output_dim, activation=self.activation_alpha)
           
            
    def feed_forward_module(self):
        dense = tf.keras.Sequential([tf.keras.layers.Flatten()])
        
        for nd in self.num_dense:
            dense.add(tf.keras.layers.Dense(nd, activation=self.dense_activation))

        return dense

    
    def call(self, x, training=False):
        x = self.init_conv(x)  
        x = self.ds_module(x)
        x = self.dense_module(x)
        
        if training:
            x = self.drop_out(x)
            
        alpha, beta, p = self.alpha(x), self.beta(x), self.p(x)
        alpha = tf.expand_dims(alpha, axis=-1)
        beta = tf.expand_dims(beta, axis=-1)
        p = tf.expand_dims(p, axis=-1)
        
        x = tf.concat((p, alpha, beta), axis=-1)

        return x 
    
    def activation_alpha(self, x):
        return 3.5 * tf.keras.activations.tanh(1/2.5 * x)


    def activation_beta(self, x):
        return 3.5 * tf.keras.activations.tanh(1/2.5 * x)
        

class BGCallWrapper:
    # post training wrapper for BGNet
    def __init__(self, bg_net, mean_mode="expected"):
        self.bg_net = bg_net
        self.mean_mode = mean_mode

    def __call__(self, x):
        params = self.bg_net(x)
        # some saved models follow the old rule
        if len(params.shape) == 3:
            p, alpha, beta = unpack_bgout(params)
            mean = BG_mean(p, alpha, beta, mode=self.mean_mode)
            if len(mean.shape) == 3 and mean.shape[-1] == 1:
                mean = tf.squeeze(mean, axis=-1)
        else:
            mean = params 
        
        return mean