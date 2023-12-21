# !pip install openml

import numpy as np
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa

from sklearn.linear_model import LinearRegression, LassoCV, RidgeClassifierCV, ElasticNetCV, LogisticRegressionCV, SGDOneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras import datasets, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose

from tqdm import tqdm

# import pymrmr

import math
import itertools
import time

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


def sean(X_train, X_test, cept=False, no_submodels=5000, num_feats_rel=0.2, order=2, prep=[], extract='ica', submodel='lin',feat_reduce_then_bagging=True, interaction_terms_then_randomize=True, desired_variance_ratio = 0.95):

    ######################################################################################
    # PRE-PROCESSING
    ######################################################################################
    def pre_process(X_train, X_test):
        # image dataset
        if len(X_train.shape) > 2:

            if 'canny' in prep:
                # A value close to 1.0 means that only the edge image is visible.
                # A value close to 0.0 means that only the original image is visible.
                # If a tuple (a, b), a random value from the range a <= x <= b will be sampled per image.
                aug = iaa.Canny(alpha=(0.75, 1.0))
                X_train = aug(images=X_train)

            if 'clahe' in prep:
                # CLAHE on all channels of the images
                aug = iaa.AllChannelsCLAHE()
                X_train = aug(images=X_train)

            if 'blur' in prep:
                # Blur each image with a gaussian kernel with a sigma of 3.0
                aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
                X_train = aug(images=X_train)

            if 'augment' in prep:
                # append augmented dataset to the existing dataset
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

                augmented_images = seq(images=X_train)
                X_train = np.concatenate((X_train, augmented_images))

            if 'gray' in prep:
                X_train = np.dot(X_train[..., :3], [0.2126 , 0.7152 , 0.0722 ])
                X_test = np.dot(X_test[..., :3], [0.2126 , 0.7152 , 0.0722 ])

            # print('flatten images')
            # Flatten the images
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            # X_train = np.reshape(X_train,(X_train.shape[0],np.prod(X_train.shape[1:])))
            # X_test = np.reshape(X_test ,(X_test.shape[0],np.prod(X_test.shape[1:])))
            # print('done flattening images. x.shape, X_test.shape : ', x.shape, X_test.shape)

        # [0,1] Normalization
        if 'norm' in prep:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if 'std' in prep:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # print("pre_process : X_train.shape X_train.shape : ", X_train.shape, X_test.shape)

        return X_train, X_test

    ######################################################################################
    # FEATURE SELECTION
    ######################################################################################
    def feature_reduction(X_train, X_test):
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

            # n_components = number of output dimensions (usually 2 for 2D visualization)
            # perplexity = a hyperparameter that controls the balance between preserving global and local structure
            # print('\n tSNE feature_reduction 1')
            tsne = TSNE(n_components=3, perplexity=30, random_state=0)
            # print('\n tSNE feature_reduction 2')
            X_train = tsne.fit_transform(X_train)
            # print('\n tSNE feature_reduction 3')
            X_test = tsne.fit_transform(X_test) # scikit tsne has no 'transform' method
            # print('\n tSNE feature_reduction 4')

        if extract == "pca":   # pca
            # pca = PCA()
            # pca.fit_transform(X_train)

            # # Calculate the cumulative explained variance
            # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            # # Determine the number of components to retain (e.g., 95% of variance)
            # num_components = np.argmax(cumulative_variance >= desired_variance_ratio) + 1

            # # Apply PCA with the selected number of components to the training and testing sets
            # pca = PCA(n_components = num_components)

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

    ######################################################################################
    # FEATURE BAGGING
    # First, standardize (mean=0, variance=1) the data; second, generate interaction terms.
    # This is to prevent multicollinearity among the new (engineered) features
    # https://www.tandfonline.com/doi/abs/10.1080/01621459.1980.10477430
    ######################################################################################
    def feature_bagging(X_train, X_test,order=order):
        # generate different set of combination of features
        # the count of features in a feature set is bounded by the order value
        np.random.seed(42)
        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        num_feats = int(math.ceil(num_feats_rel*X_train.shape[1]))

        if interaction_terms_then_randomize:
            # Build interaction terms using PolynomialFeatures
            poly = PolynomialFeatures(degree=order, include_bias=False)
            X_train_interaction_terms = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names_out(X_train.columns))
            X_test_interaction_terms = pd.DataFrame(poly.transform(X_test), columns=poly.get_feature_names_out(X_test.columns))
            # print('interaction_terms_then_randomize : X_train_interaction_terms ', X_train_interaction_terms)

            # Select a random subset of features (excluding the interaction terms)
            selected_features = np.random.choice(X_train.columns, size=num_feats, replace=False)
            # print('interaction_terms_then_randomize : selected_features ', selected_features)

            # Extract the selected features from the interaction terms
            X_train = X_train_interaction_terms[selected_features]
            X_test = X_test_interaction_terms[selected_features]

        else:
            # Select a random subset of features
            selected_features = np.random.choice(X_train.columns, size=num_feats, replace=False)

            # Extract the selected features from the dataset
            X_train_selected_features = X_train[selected_features]
            X_test_selected_features = X_test[selected_features]

            # Build interaction terms using PolynomialFeatures
            poly = PolynomialFeatures(degree=order, include_bias=False)
            X_train = pd.DataFrame(poly.fit_transform(X_train_selected_features), columns=poly.get_feature_names_out(selected_features))
            X_test = pd.DataFrame(poly.transform(X_test_selected_features), columns=poly.get_feature_names_out(selected_features))

            # # Select a random subset of features (excluding the interaction terms)
            # selected_features = np.random.choice(X_train.columns, size=num_feats, replace=False)

            # # Extract the selected features from the interaction terms
            # X_train = X_train[selected_features]
            # X_test = X_test[selected_features]

        # print("feature_bagging X_train.shape  X_test.shape: ", X_train.shape, X_test.shape)

        return X_train, X_test


    ######################################################################################
    # ENSEMBLE
    ######################################################################################
    def one_model(X_train, X_test):
        # print('one_model')

        # eqn. 2 from the DEAN paper
        goal=np.ones(len(X_train))

        if submodel ==  "lin":
            cv = LinearRegression(fit_intercept=cept).fit(X_train, goal)
        if submodel ==  "lasso":
            cv = LassoCV(fit_intercept=cept).fit(X_train, goal)
        if submodel ==  "elastic":
            cv = ElasticNetCV(fit_intercept=cept).fit(X_train, goal)
        if submodel ==  "svm":
            cv = SGDOneClassSVM(fit_intercept=cept).fit(X_train, goal)

        # eqn. 4 from the DEAN paper
        meanv = np.mean(cv.predict(X_train))
        # print("meanv=np.mean(cv.predict(x)) : ", meanv)
        pred = np.square(cv.predict(X_test)-meanv)
        # print("\n pred=np.square(cv.predict(tx)-meanv) : ", pred)

        return pred

    # Set the maximum computation time (in seconds)
    max_computation_time = 600  # 10 minutes
    start_time = time.time()
    
    X_train, X_test = pre_process(X_train, X_test)
    if feat_reduce_then_bagging:
        X_train, X_test = feature_reduction(X_train, X_test)
        X_train, X_test = feature_bagging(X_train, X_test)
    else:
        X_train, X_test = feature_bagging(X_train, X_test)
        X_train, X_test = feature_reduction(X_train, X_test)

    scores=[]
    ensembles_executed = 0

    elapsed_time = time.time() - start_time
    if elapsed_time > max_computation_time:
        print("Feature reduction: Time limit reached. Exiting.")
        return np.zeros(X_test.shape[0]), ensembles_executed

    else:
        for i in tqdm(range(no_submodels)):
            pred = one_model(X_train, X_test)
            scores.append(pred)

            ensembles_executed += 1

            elapsed_time = time.time() - start_time
            if elapsed_time > max_computation_time:
                print("Ensemble: Time limit reached. Exiting loop.")
                break

        #features=list(generate_subsets(list(range(x.shape[1])),num_feats))

        scores=np.array(scores)

        # print('Mean = {}, Variance = {}'.format(np.mean(scores,axis=0),np.var(scores,axis=0)))

        # eqn. 5 from the DEAN paper
        return np.mean(scores,axis=0), ensembles_executed


# # executes only when run directly, not when this file is imported into another python file
# if __name__ == '__main__':
#     sean()