# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import RMSprop
import keras.backend as K

img_size = 64
latent_size = 512
batch_size = 16
iterations = 50000

assert log2(img_size).is_integer, 'input image size must be a power of 2'
n_layers = int(log2(img_size))

d_loss = []
g_loss = []
gp_loss = []

def noise(n, latent_size):
    return np.random.normal(0.0, 1.0, size=[n, latent_size]).astype(float)

def noise_list(n, n_layers, latent_size):
    return [noise(n, latent_size)] * n_layers

# mixing regularization
def mixed_list(n, layers, latent_size):
    break_point = int(random() * layers)
    return noise_list(n, break_point, latent_size) + noise_list(n, layers - break_point, latent_size)

def gradient_penalty(img, sample, weight):
    # imgs are true images while samples are generated images
    gradients = K.gradients(img, sample)
    gradient_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradients_sqr_sum)

# Adaptive Instance Normalization
def AdaIN(img):
    mean = K.mean(img[0], axis=[0, 1], keepdims=True)
    std = K.std(img[0], axis=[0, 1], keepdims=True)
    out = (img[0] - mean) / std
    
    pool_shape = [-1, 1, 1, out.shape[-1]]
    scale = K.reshape(x[1], pool_shape)
    bias = K.reshape(x[2], pool_shape)
    
    return out * scale + bias

def g_block(inp_tensor, latent_vector, filters):
    scale = Dense(filters)(latent_vector)
    bias = Dense(filters)(latent_vector)
    
    out = UpSampling2D()(input_tensor)
    out = Conv2D(filters, 3, padding='same')(out)
    out = Lambda(AdaIN)([out, scale, bias])
    out = LeakyReLU(alpha=0.3)(out)
    
    return out

def d_block(inp_tensor, filters):
    out = Conv2D(filters, 3, padding='same')(inp_tensor)
    out = LeakyReLU(alpha=0.2)(out)
    out = Conv2D(filters, 3, padding='same')(out)
    out = LeakyReLU(alpha=0.2)(out)
    out = AveragePooling2D(out)
    
    return out

class StyleGAN():
    
    def __init__(self, steps=1, lr=0.0001, latent_size=latent_size, n_layers=n_layers):
        self.latent_size = latent_size
        self.steps = 1
        self.lr = lr
        
    def build_generator(self):
        latent_input = Input(shape=[latent_size])
        
        # latent mapping network
        latent = Dense(64)(latent_input)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        
        out = Dense(4*4*64, activation='relu')(inp)
        out = Reshape([4, 4, 64])(out)
        
        # 4*4*64
        out = g_block(inp, latent, 64)
        # 8*8*64
        out = g_block(inp, latent, 32)
        # 16*16*32
        out = g_block(inp, latent, 32)
        # 32*32*16
        out = g_block(inp, latent, 16)
        # 64*64*8
        image_output = Conv2D(3, 1, padding='same', activation='sigmoid')(out)
        
        generator_model = Model(inputs=latent_input, outputs=image_output)
        
        return generator_model