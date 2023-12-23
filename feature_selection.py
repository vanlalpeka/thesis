import numpy as np
import pandas as pd
import math
import itertools

from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose

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

        # self.encoder = tf.keras.Sequential([
        #   layers.Input(shape=shape),
        #   layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        #   layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        # self.decoder = tf.keras.Sequential([
        #   layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        #   layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        #   layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)). One could vary the
        # standard deviation by multiplying the interval with appropriate value.
        # Here we initialize the weights with mean 0 and standard deviation 0.1.
        # Reference: Understanding the difficulty of training deep feedforward
        # neural networks by Xavier Glorot and Yoshua Bengio
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
                low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                            size=(num_visible, num_hidden)))


        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs = 1000, learning_rate = 0.1, patience = 3, wait = 0, best = float('inf')):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:,0] = 1 # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.

            # The visible layer is Gaussian
            neg_visible_probs = neg_visible_probs + np.random.normal(0, 0.1, neg_visible_probs.shape)

            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

            # early stopping
            wait += 1
            if error < best:
                best = error
                wait = 0

            if wait >= patience:
                break

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:,1:]
        return hidden_states

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

######################################################################################
# FEATURE SELECTION
######################################################################################
def feature_selection(X_train, X_test, num_feats_rel, extract):
    # print(f'feature_selection: {num_feats_rel}  {extract}')
    n_components = int(math.ceil(num_feats_rel*X_train.shape[1]))
    max_epochs = int(np.sqrt(X_train.shape[1]) * 10)

    if extract == "rbm":   # ZCA + RBM
        # Calculate the mean of each of the columns
        mean_X_train = np.mean(X_train, axis=0)
        mean_X_test = np.mean(X_test, axis=0)

        # Center the data by subtracting the mean
        centered_X_train = X_train - mean_X_train
        centered_X_test = X_test - mean_X_test

        # Calculate the covariance matrix
        covariance_matrix = np.cov(centered_X_train, rowvar=False)

        # Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Calculate the ZCA components
        zca_components = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

        # Apply ZCA whitening to the data
        zca_whitened_X_train = np.dot(centered_X_train, zca_components)
        zca_whitened_X_test = np.dot(centered_X_test, zca_components)

        # RBM
        r = RBM(num_visible = zca_whitened_X_train.shape[1], num_hidden = n_components)
        r.train(zca_whitened_X_train, max_epochs = max_epochs)
        # r.train(zca_whitened_X_train, max_epochs = 25)
        X_train = r.run_visible(zca_whitened_X_train)
        X_test = r.run_visible(zca_whitened_X_test)


    if extract == "tsne":   # tSNE
        # FEATURE SELECTION to reduce training duration
        if X_train.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
        tsne = TSNE(n_components=3, perplexity=30, random_state=0)
        X_train = tsne.fit_transform(X_train)
        X_test = tsne.fit_transform(X_test) # scikit tsne has no 'transform' method

    if extract == "pca":  
        pca = PCA(n_components = n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    if extract == "ica":   # ica
        fastICA = FastICA(n_components = n_components, whiten='unit-variance')
        X_train = fastICA.fit_transform(X_train)
        X_test = fastICA.transform(X_test)

    if extract == "nmf":   # nmf
        # NMF is a non-convex optimization problem, so the results may vary with different initializations
        nmf = NMF(n_components = n_components, init='random', random_state=0)
        X_train = nmf.fit_transform(X_train)
        X_test = nmf.fit_transform(X_test) # scikit has no separate transform() function for NMF
        # X_train_features = nmf.components_

    if extract == "ae":   # Autoencoder
        # Basic Autoencoder
        autoencoder = Autoencoder(n_components, X_train.shape[1:])

        # # Convolution Autoencoder
        # autoencoder = ConvAutoencoder()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder.fit(X_train, X_train,
                        epochs=max_epochs,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[early_stopping]
                        )

        X_train = autoencoder.encoder(X_train).numpy()
        X_test = autoencoder.encoder(X_test).numpy()

    # print(f"feature_reduction {extract} X_train.shape : {X_train.shape},  X_test.shape: {X_test.shape}")

    return X_train, X_test