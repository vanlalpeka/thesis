# !pip install openml

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LassoCV, RidgeClassifierCV, ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.models import Model

from tqdm import tqdm

# import pymrmr

import math
import itertools

# # for RBM
# from pydbm.activation.logistic_function import LogisticFunction
# from pydbm.activation.tanh_function import TanhFunction
# from pydbm.optimization.optparams.sgd import SGD
# from pydbm.dbm.restrictedboltzmannmachines.rt_rbm import RTRBM

def generate_subsets(sequence, leng=3):
    # print('generate_subsets')
    subsets = []
    n = len(sequence)

    for r in range(n+1):
        for zw in itertools.combinations(sequence, r):
            if len(zw) == leng or leng is None:
                yield zw


def all_features(*args, leng=3):
    # print('all_features')
    F=int(args[0].shape[1])
    for seq in generate_subsets(list(range(F)),leng):
        yield [xx[:,seq] for xx in args]

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
          layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(28, 28, 1)),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded






# Calculates ZCA components from xx. Applies that to both xx and txx
def ZCA(xx, txx):
    # Calculate the mean of each of the columns
    mean_xx = np.mean(xx, axis=0)
    mean_txx = np.mean(txx, axis=0)

    # Center the data by subtracting the mean
    centered_xx = xx - mean_xx
    centered_txx = txx - mean_txx

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_xx, rowvar=False)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Calculate the ZCA components
    zca_components = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

    # Apply ZCA whitening to the data
    zca_whitened_xx = np.dot(centered_xx, zca_components)
    zca_whitened_txx = np.dot(centered_txx, zca_components)

    return zca_whitened_xx, zca_whitened_txx


def sean(x, tx, cept=False, no_submodels=5000, num_feats_rel=0.2, num_feats_abs=10, order=2, relative_feats=True, min_feats=3, prep=[], extract='ica', submodel='lin', feat_type='manual'):

    # print('In SEAN')
    # print('Params: number of submodels {} \n Linear? {} \n Feature Selector? {} \n Relative features? {} \n Altnorm? {} \n Min Features? {} \n'.format(no_submodels, justlin,projections,relative_feats,altnorm,min_feats))
    ######################################################################################
    # PRE-PROCESSING
    ######################################################################################
    def pre_process(x,tx):
        # image dataset
        if len(x.shape) > 2:
            if 'gray' in prep:
                x = np.dot(x[..., :3], [0.2126 , 0.7152 , 0.0722 ])
                tx = np.dot(tx[..., :3], [0.2126 , 0.7152 , 0.0722 ])

            # print('flatten images')
            # Flatten the images
            # x = xx.reshape(xx.shape[0], -1)
            # tx = txx.reshape(txx.shape[0], -1)

            x = np.reshape(x,(x.shape[0],np.prod(x.shape[1:])))
            tx = np.reshape(tx ,(tx.shape[0],np.prod(tx.shape[1:])))
            # print('done flattening images. x.shape, tx.shape : ', x.shape, tx.shape)

        # [0,1] Normalization
        if 'norm' in prep:
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            tx = scaler.transform(tx)

        # [0.5,1] Normalization
        if 'altnorm' in prep:
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            tx = scaler.transform(tx)
            x=(1+x)/2
            tx=(1+tx)/2

        if 'std' in prep:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            tx = scaler.transform(tx)

            # print("Less than zero:", x[x<0])

        if 'robust' in prep:
            scaler = RobustScaler()
            x = scaler.fit_transform(x)
            tx = scaler.transform(tx)

        print("pre_process. x shape:",x.shape)

        return x,tx

    ######################################################################################
    # FEATURE SELECTION
    ######################################################################################
    def feature_selection(xx, txx):
        print("feature_selection xx.shape : ", xx.shape)
        n_components = int(math.ceil(num_feats_rel*xx.shape[1]))

        match extract:
            case "rbm":   # ZCA +RBM
                # ZCA(): Calculates ZCA components from xx. Applies that to both xx and txx
                x, txx = ZCA(xx, txx)
                print('rbm 1')

                rbm = RBM(num_visible=xx.shape[1], num_hidden=n_components)
                print('rbm 2')
                x = rbm.train(x, learning_rate=0.1, epochs=50, batch_size=100)
                print('rbm 3')
                tx = rbm.train(tx, learning_rate=0.1, epochs=50, batch_size=100)
                print('rbm 4')

            case "tsne":   # tSNE
                # Dimensionality SELECTION to reduce training duration
                if xx.shape[1] > n_components:
                    pca = PCA(n_components=n_components)
                    xx = pca.fit_transform(xx)
                    txx = pca.transform(txx)

                # n_components = number of output dimensions (usually 2 for 2D visualization)
                # perplexity = a hyperparameter that controls the balance between preserving global and local structure
                # print('\n tSNE feature_selection 1')
                tsne = TSNE(n_components=2, perplexity=30, random_state=0)
                # print('\n tSNE feature_selection 2')
                x = tsne.fit_transform(xx)
                # print('\n tSNE feature_selection 3')
                tx = tsne.fit_transform(txx) # there is no 'transform' method
                # print('\n tSNE feature_selection 4')

            case "pca":   # PCA
                pca = PCA(n_components = n_components)
                x = pca.fit_transform(xx)
                tx = pca.transform(txx)

            case "ica":   # ICA
                fastICA = FastICA(n_components = n_components, whiten='unit-variance')
                x = fastICA.fit_transform(xx)
                tx = fastICA.transform(txx)

            case "nmf":   # NMF
                # NMF is a non-convex optimization problem, so the results may vary with different initializations
                nmf = NMF(n_components = n_components, init='random', random_state=0)
                x = nmf.fit_transform(xx)
                tx = nmf.fit_transform(txx) # there is no separate transform method
                # xx_features = nmf.components_

            case "ae":   # Autoencoder
                print('feature_selection AE xx.shape : ', xx.shape)
                shape = xx.shape[1:]
                latent_dim = int(math.ceil(num_feats_rel*shape[0]))

                # Basic Autoencoder
                autoencoder = Autoencoder(latent_dim, shape)

                # # Convolution Autoencoder
                # autoencoder = ConvAutoencoder()

                autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
                autoencoder.fit(xx, xx,
                                epochs=10,
                                shuffle=True,
                                validation_data=(txx, txx))

                x = autoencoder.encoder(xx).numpy()
                tx = autoencoder.encoder(txx).numpy()

            # case "mrmr":   # MRMR feature reduction
            #     pymrmr.mRMR(xx, 'MID',6)


            case _:
                x = xx
                tx = txx

        return x, tx

    ######################################################################################
    # FEATURE ENGINEERING
    # First, standardize (mean=0, variance=1) the data; second, generate interaction terms.
    # This is to prevent multicollinearity among the new (engineered) features
    # https://www.tandfonline.com/doi/abs/10.1080/01621459.1980.10477430
    ######################################################################################
    def feature_engineering(x,tx,order=order):
        print('feature_engineering')

        match feat_type:
            case 'mrmr':
                pass

            case 'manual':
                #all ordered subsequences up to count of length up to order
                orderings = list(itertools.chain.from_iterable(itertools.combinations(range(x.shape[1]),i) for i in range(1,order+1)))

                # print(list(orderings))

                if relative_feats:
                    num_feats = int(math.ceil(num_feats_rel*len(orderings)))
                else:
                    num_feats = num_feats_abs

                if num_feats < min_feats:
                    num_feats = min_feats

                # print(num_feats)

                # update order
                xx,txx=[],[]
                for order in orderings:
                    # print('update_order for-loop: order =={}'.format(order))
                    xx.append(np.prod([x[:,i] for i in order],axis=0))
                    txx.append(np.prod([tx[:,i] for i in order],axis=0))
                # print('update order: xx.shape {}'.format(np.array(xx).shape))

                x = np.array(xx).T
                tx = np.array(txx).T

                if num_feats > x.shape[1]:
                    num_feats = x.shape[1]
                feats = np.random.choice(range(x.shape[1]), num_feats, replace=False)
                return x[:,feats],tx[:,feats]

            case _:
                raise Exception('Invalid feat_type')

    # no_of_covariates=int(np.ceil(x.shape[1]/10))#3


    ######################################################################################
    # ENSEMBLE
    ######################################################################################
    def one_model(xx,txx):
        # print('one_model')

        x, tx = pre_process(xx,txx)
        x, tx = feature_selection(x,tx)
        x, tx = feature_engineering(x,tx)

        # eqn. 2 from the DEAN paper
        goal=np.ones(len(x))

        match submodel:
            case "lin":
                cv = LinearRegression(fit_intercept=cept).fit(x,goal)
            case "lasso":
                cv = LassoCV(fit_intercept=False).fit(x,goal)
            case "ridge":
                cv = RidgeClassifierCV(fit_intercept=False).fit(x,goal)
            case "elastic":
                cv = ElasticNetCV(fit_intercept=False).fit(x,goal)
            case "log":
                cv = LogisticRegressionCV(fit_intercept=False).fit(x,goal)
            # case "dtree":
            #     cv = DecisionTreeClassifier().fit(x,goal)
            case _:
                raise Exception("Please specify a submodel type")

        # eqn. 4 from the DEAN paper
        meanv=np.mean(cv.predict(x))
        print("meanv=np.mean(cv.predict(x)) : ", meanv)
        pred=np.square(cv.predict(tx)-meanv)
        print("pred=np.square(cv.predict(tx)-meanv) : ", pred)

        return pred

    scores=[]
    for i in tqdm(range(no_submodels)):
        # print('sub model {} of {}'.format(i,no_submodels))
        # print('Selected features of X : {}'.format(np.array(xx).shape))
        pred=one_model(x,tx)
        scores.append(pred)

    #features=list(generate_subsets(list(range(x.shape[1])),num_feats))

    scores=np.array(scores)

    # print('Mean = {}, Variance = {}'.format(np.mean(scores,axis=0),np.var(scores,axis=0)))

    # eqn. 5 from the DEAN paper
    return np.mean(scores,axis=0)


# # executes only when ran directly, not when this file is imported into another python file
# if __name__ == '__main__':
#     sean()