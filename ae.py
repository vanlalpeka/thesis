import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose

from tqdm import tqdm

# import pymrmr

import math
import itertools


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          Flatten(),
          Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
          Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded