# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import RMSprop
import keras.backend as K

img_size = 1024
latent_size = 512
batch_size = 16
iterations = 50000

assert log2(img_size).is_integer, 'input image size must be a power of 2'
n_layers = int(log2(img_size))

d_loss = []
g_loss = []
gp_loss = []

def gradient_penalty(imgs, sample, weight):
    # imgs are true images while samples are generated images
    gradients = K.gradients(imgs, sample)
    gradient_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradients_sqr_sum)

