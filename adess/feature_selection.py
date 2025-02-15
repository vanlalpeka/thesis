import numpy as np
# import pandas as pd
import math

from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.manifold import TSNE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping

from adess.autoencoder import Autoencoder
from adess.rbm import RBM


def feature_selection(X_train, X_test, feat_sel_percent, max_feats, extract):
    """
    X_train and X_test are ndarray of the train and the test sets.
    Image datasets are already flattened in the pre_process() function.

    feat_sel_percent: The percentage of features to select e.g. 0.2 means select 20% of the original features.

    extract: A feature selection method. Options are ica, pca, nmf, rbm, ae, tsne.
    """
    # print(f'feature_selection: {feat_sel_percent}  {extract}')

    # Limit the number of features to max_feats
    n_components = int(math.ceil(feat_sel_percent*X_train.shape[1]))
    if n_components > max_feats:
        n_components = max_feats

    max_epochs = int(np.sqrt(X_train.shape[1]) * 10)

    if extract == 'none':
        return X_train, X_test

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
        # additional feature selection to reduce training time
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
                        callbacks=[early_stopping],
                        verbose=0,
                        )

        X_train = autoencoder.encoder(X_train).numpy()
        X_test = autoencoder.encoder(X_test).numpy()

    # print(f"feature_reduction {extract} X_train.shape : {X_train.shape},  X_test.shape: {X_test.shape}")

    return X_train, X_test


if __name__ == '__main__':
    feature_selection()