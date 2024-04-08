import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape

        # Basic Autoencoder
        self.encoder = tf.keras.Sequential([
          Flatten(),
          Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
          Reshape(shape)
        ])

        # self.encoder = tf.keras.Sequential([
        #   Flatten(),
        #   Dense(units=shape[0] // 2, activation='relu'),
        #   Dense(units=shape[0] // 4, activation='relu'),
        #   Dense(units=shape[0] // 8, activation='relu'),
        #   Dense(units=shape[0] // 16, activation='relu'),
        # ])
        # self.decoder = tf.keras.Sequential([
        #   Dense(units=shape[0] // 8, activation='relu'),
        #   Dense(units=shape[0] // 4, activation='relu'),
        #   Dense(units=shape[0] // 2, activation='relu'),
        #   Dense(units=shape[0], activation='sigmoid'),
        #   Reshape(shape)
        # ])

        #  # Convolution Autoencoder
        # self.encoder = tf.keras.Sequential([
        #   Input(shape=shape),
        #   Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        #   Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        # self.decoder = tf.keras.Sequential([
        #   Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        #   Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        #   Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded