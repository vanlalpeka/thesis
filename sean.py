# !pip install openml

import numpy as np
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa

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

# https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
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

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
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



def sean(x, tx, cept=False, no_submodels=5000, num_feats_rel=0.2, num_feats_abs=10, order=2, relative_feats=True, min_feats=3, prep=[], extract='ica', submodel='lin', feat_type='manual', epochs=10):

    # print('In SEAN')
    # print('Params: number of submodels {} \n Linear? {} \n Feature Selector? {} \n Relative features? {} \n Altnorm? {} \n Min Features? {} \n'.format(no_submodels, justlin,projections,relative_feats,altnorm,min_feats))
    ######################################################################################
    # PRE-PROCESSING
    ######################################################################################
    def pre_process(x,tx):
        print('pre_process')
        # image dataset
        if len(x.shape) > 2:

            if 'canny' in prep:
                # A value close to 1.0 means that only the edge image is visible.
                # A value close to 0.0 means that only the original image is visible.
                # If a tuple (a, b), a random value from the range a <= x <= b will be sampled per image.
                aug = iaa.Canny(alpha=(0.75, 1.0))
                x = aug(images=x)

            if 'clahe' in prep:
                # CLAHE on all channels of the images
                aug = iaa.AllChannelsCLAHE()
                x = aug(images=x)

            if 'blur' in prep:
                # Blur each image with a gaussian kernel with a sigma of 3.0
                aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
                x = aug(images=x)

            if 'augment' in prep:
                ia.seed(1)

                seq = iaa.Sequential([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # # Small gaussian blur with random sigma between 0 and 0.5.
                    # # But we only blur about 50% of all images.
                    # iaa.Sometimes(
                    #     0.5,
                    #     iaa.GaussianBlur(sigma=(0, 0.5))
                    # ),
                    # # Strengthen or weaken the contrast in each image.
                    # iaa.LinearContrast((0.75, 1.5)),
                    # # Add gaussian noise.
                    # # For 50% of all images, we sample the noise once per pixel.
                    # # For the other 50% of all images, we sample the noise per pixel AND
                    # # channel. This can change the color (not only brightness) of the
                    # # pixels.
                    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # # Make some images brighter and some darker.
                    # # In 20% of all cases, we sample the multiplier once per channel,
                    # # which can end up changing the color of the images.
                    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-25, 25),
                        shear=(-8, 8)
                    )
                ], random_order=True) # apply augmenters in random order

                augmented_images = seq(images=x)
                x = np.concatenate((x,augmented_images))

            if 'gray' in prep:
                x = np.dot(x[..., :3], [0.2126 , 0.7152 , 0.0722 ])
                tx = np.dot(tx[..., :3], [0.2126 , 0.7152 , 0.0722 ])

            # print('flatten images')
            # Flatten the images
            # x = x.reshape(x.shape[0], -1)
            # tx = tx.reshape(tx.shape[0], -1)

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

        # print("pre_process: x.shape:",x.shape)

        return x,tx

    ######################################################################################
    # FEATURE SELECTION
    ######################################################################################
    def feature_selection(x, tx):
        print("feature_selection x.shape : ", x.shape)
        n_components = int(math.ceil(num_feats_rel*x.shape[1]))

        match extract:
            case "rbm":   # ZCA + RBM

                # Calculate the mean of each of the columns
                mean_x = np.mean(x, axis=0)
                mean_tx = np.mean(tx, axis=0)

                # Center the data by subtracting the mean
                centered_x = x - mean_x
                centered_tx = tx - mean_tx

                # Calculate the covariance matrix
                covariance_matrix = np.cov(centered_x, rowvar=False)

                # Perform eigenvalue decomposition on the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

                # Calculate the ZCA components
                zca_components = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

                # Apply ZCA whitening to the data
                zca_whitened_x = np.dot(centered_x, zca_components)
                zca_whitened_tx = np.dot(centered_tx, zca_components)

                # RBM
                r = RBM(num_visible = zca_whitened_x.shape[1], num_hidden = n_components)
                r.train(zca_whitened_x, max_epochs = epochs)
                x = r.run_visible(zca_whitened_x)
                tx = r.run_visible(zca_whitened_tx)

            case "tsne":   # tSNE
                # FEATURE SELECTION to reduce training duration
                if x.shape[1] > n_components:
                    pca = PCA(n_components=n_components)
                    x = pca.fit_transform(x)
                    tx = pca.transform(tx)

                # n_components = number of output dimensions (usually 2 for 2D visualization)
                # perplexity = a hyperparameter that controls the balance between preserving global and local structure
                # print('\n tSNE feature_selection 1')
                tsne = TSNE(n_components=2, perplexity=30, random_state=0)
                # print('\n tSNE feature_selection 2')
                x = tsne.fit_transform(x)
                # print('\n tSNE feature_selection 3')
                tx = tsne.fit_transform(tx) # scikit tsne has no 'transform' method
                # print('\n tSNE feature_selection 4')

            case "pca":   # PCA
                pca = PCA(n_components = n_components)
                x = pca.fit_transform(x)
                tx = pca.transform(tx)

            case "ica":   # ICA
                fastICA = FastICA(n_components = n_components, whiten='unit-variance')
                x = fastICA.fit_transform(x)
                tx = fastICA.transform(tx)

            case "nmf":   # NMF
                # NMF is a non-convex optimization problem, so the results may vary with different initializations
                nmf = NMF(n_components = n_components, init='random', random_state=0)
                x = nmf.fit_transform(x)
                tx = nmf.fit_transform(tx) # scikit has no separate transform() function for NMF
                # x_features = nmf.components_

            case "ae":   # Autoencoder

                # Basic Autoencoder
                autoencoder = Autoencoder(n_components, x.shape[1:])

                # # Convolution Autoencoder
                # autoencoder = ConvAutoencoder()

                autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
                autoencoder.fit(x, x,
                                epochs=epochs,
                                shuffle=True,
                                validation_data=(tx, tx))

                x = autoencoder.encoder(x).numpy()
                tx = autoencoder.encoder(tx).numpy()

            # case "mrmr":   # MRMR feature reduction
            #     pymrmr.mRMR(x, 'MID',6)


            case _:
                pass

        return x, tx

    ######################################################################################
    # FEATURE ENGINEERING
    # First, standardize (mean=0, variance=1) the data; second, generate interaction terms.
    # This is to prevent multicollinearity among the new (engineered) features
    # https://www.tandfonline.com/doi/abs/10.1080/01621459.1980.10477430
    ######################################################################################
    def feature_engineering(x,tx,order=order):
        print(f'feature_engineering x.shape {x.shape} ; tx.shape {tx.shape}')

        match feat_type:
            case 'mrmr':
                pass

            case 'manual':
                print('Manual feature bagging')
                # generate different set of combination of features
                # the count of features in a feature set is bounded by the order value
                orderings = list(itertools.chain.from_iterable(itertools.combinations(range(x.shape[1]),i) for i in range(1,order+1)))

                if relative_feats:
                    num_feats = int(math.ceil(num_feats_rel*len(orderings)))
                else:
                    num_feats = num_feats_abs

                if num_feats < min_feats:
                    num_feats = min_feats

                print(num_feats)

                # multiply the values of the features in the feature set
                xx,txx=[],[]
                for order in orderings:
                    # print('update_order for-loop: order =={}'.format(order))
                    xx.append(np.prod([x[:,i] for i in order],axis=0))
                    txx.append(np.prod([tx[:,i] for i in order],axis=0))
                # print('update order: x.shape {}'.format(np.array(x).shape))

                x = np.array(xx).T
                tx = np.array(txx).T

                if num_feats > x.shape[1]:
                    num_feats = x.shape[1]

                # select random features from the interaction terms
                feats = np.random.choice(range(x.shape[1]), num_feats, replace=False)
                return x[:,feats],tx[:,feats]

            case _:
                raise Exception('Invalid feat_type')

    ######################################################################################
    # ENSEMBLE
    ######################################################################################
    def one_model(x,tx):
        print('one_model')
        # eqn. 2 from the DEAN paper
        goal=np.ones(len(x))

        match submodel:
            case "lin":
                cv = LinearRegression(fit_intercept=cept).fit(x,goal)
            case "lasso":
                cv = LassoCV(fit_intercept=cept).fit(x,goal)
            case "ridge":
                cv = RidgeClassifierCV(fit_intercept=cept).fit(x,goal)
            case "elastic":
                cv = ElasticNetCV(fit_intercept=cept).fit(x,goal)
            case "log":
                cv = LogisticRegressionCV(fit_intercept=cept).fit(x,goal)
            # case "dtree":
            #     cv = DecisionTreeClassifier().fit(x,goal)
            case _:
                raise Exception("Please specify a submodel type")

        # eqn. 4 from the DEAN paper
        meanv=np.mean(cv.predict(x))
        # print("meanv=np.mean(cv.predict(x)) : ", meanv)
        pred=np.square(cv.predict(tx)-meanv)
        # print("\n pred=np.square(cv.predict(tx)-meanv) : ", pred)

        return pred

    print(f"prep is {prep}")
    x, tx = pre_process(x,tx)
    x, tx = feature_selection(x,tx)
    x, tx = feature_engineering(x,tx)

    scores=[]
    for i in tqdm(range(no_submodels)):
        # print('sub model {} of {}'.format(i,no_submodels))
        # print('Selected features of X : {}'.format(np.array(x).shape))
        pred=one_model(x,tx)
        scores.append(pred)

    #features=list(generate_subsets(list(range(x.shape[1])),num_feats))

    scores=np.array(scores)

    # print('Mean = {}, Variance = {}'.format(np.mean(scores,axis=0),np.var(scores,axis=0)))

    # eqn. 5 from the DEAN paper
    return np.mean(scores,axis=0)


# # executes only when run directly, not when this file is imported into another python file
# if __name__ == '__main__':
#     sean()