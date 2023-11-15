# import importlib.util

# if importlib.util.find_spec('pydbm') is None:
#     !pip install pydbm
# else:
#     print('pydbm is already installed')

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from tqdm import tqdm

import math
import itertools

# # for RBM
# from pydbm.activation.logistic_function import LogisticFunction
# from pydbm.activation.tanh_function import TanhFunction
# from pydbm.optimization.optparams.sgd import SGD
# from pydbm.optimization.optparams.adam import Adam
# from pydbm.dbm.restrictedboltzmannmachines.rt_rbm import RTRBM


def generate_subsets(sequence, leng=3):
    print('generate_subsets')
    subsets = []
    n = len(sequence)

    for r in range(n+1):
        for zw in itertools.combinations(sequence, r):
            if len(zw) == leng or leng is None:
                yield zw


def all_features(*args, leng=3):
    print('all_features')
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

# def RBM(xx,txx,num_feats_rel)
#     learning_rate = 0.01
#     training_epochs = 50
#     batch_size = 100
#     n_hidden = int(math.ceil(num_feats_rel*xx.shape[1]))
#     n_visible = xx.shape[1]

#     X = tf.placeholder("float", [None, n_visible], name='X')
#     X_noise = tf.placeholder("float", [None, n_visible], name='X_noise')

#     # RBM parameters
#     W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name='W')
#     b_visible = tf.Variable(tf.zeros([n_visible]), name='b_visible')
#     b_hidden = tf.Variable(tf.zeros([n_hidden]), name='b_hidden')

#     # activation functions
#     def sigmoid(x):
#         return 1. / (1. + tf.exp(-x))

#     def sample_prob(probs):
#         return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

#     # the positive phase, which calculates the expected value of the hidden layer activations given the input data
#     h_prob = sigmoid(tf.matmul(X_noise, W) + b_hidden)
#     h_state = sample_prob(h_prob)

#     # the negative phase, which calculates the expected value of the visible layer activations given the hidden layer activations
#     v_prob = sigmoid(tf.matmul(h_state, tf.transpose(W)) + b_visible)
#     v_state = sample_prob(v_prob)

#     # the reconstruction error, which measures the difference between the input data and the reconstructed data
#     err = tf.reduce_mean(tf.square(X - v_state))

#     # the training operation, which updates the RBM parameters to minimize the reconstruction error.
#     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(err)

#     # Train the RBM
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#       sess.run(init)
#       for epoch in range(training_epochs):
#         total_batch = int(xx.shape[0] / batch_size)
#         for i in range(total_batch):
#           batch_xs, _ = mnist.train.next_batch(batch_size)
#           batch_xs_noise = batch_xs + 0.3*np.random.randn(*batch_xs.shape)
#           batch_xs_noise = np.clip(batch_xs_noise, 0., 1.)
#           batch_xs = np.clip(batch_xs, 0., 1.)
#           _, c = sess
#       sess.run(train_op, feed_dict={X: batch_xs, X_noise: batch_xs_noise})
#       print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
#     print("RBM training finished!")

#     # Test the RBM on the test data
#     batch_xs, _ = mnist.test.next_batch(10)
#     batch_xs_noise = batch_xs + 0.3*np.random.randn(*batch_xs.shape)
#     batch_xs_noise = np.clip(batch_xs_noise, 0., 1.)
#     batch_xs = np.clip(batch_xs, 0., 1.)
#     err_test = sess.run(err, feed_dict={X: batch_xs, X_noise: batch_xs_noise})
#     print("RBM test error:", err_test)


def ZCA(df):
    # Calculate the mean of each of the columns
    mean = np.mean(df, axis=0)

    # Center the data by subtracting the mean
    centered_data = df - mean

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Calculate the ZCA components
    zca_components = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

    # Apply ZCA whitening to the data
    zca_whitened_data = np.dot(centered_data, zca_components)

    return zca_whitened_data


def sean(x,tx,cept=False,justlin=True,projections=False,no_submodels=5000,num_feats_rel=0.2,num_feats_abs=10,order=2,relative_feats=True,altnorm=True,min_feats=3, prep=0, extract=0):
    #use intercept? No ensemble learning with an intercept
    #use linear regression?
    #feature selector
    #number of submodels
    #number of features
    #taylor order of features
    #relative num_feats?
    #normalize between 0.5 and 1 instead 0 and 1
    #minimum number of feats
    # preprocessing
    # Feature extractor

    # print('In SEAN')
    # print('Params: number of submodels {} \n Linear? {} \n Feature Selector? {} \n Relative features? {} \n Altnorm? {} \n Min Features? {} \n'.format(no_submodels, justlin,projections,relative_feats,altnorm,min_feats))
    ######################################################################################
    # PRE-PROCESSING
    ######################################################################################
    def pre_process(xx,txx):
        if prep == 1:        # [0,1] Normalization
            scaler = MinMaxScaler()
            x = scaler.fit_transform(xx)
            # x = pd.DataFrame(x, columns=xx.columns)
            # tx = MinMaxScaler().fit_transform(txx)
            tx = scaler.transform(txx)
            # tx = pd.DataFrame(tx, columns=txx.columns)

            if altnorm: # [0.5,1] Normalization
              x=(1+x)/2
              tx=(1+tx)/2

        return x,tx

    ######################################################################################
    # FEATURE EXTRACTION
    ######################################################################################
    def extract_features(xx, txx):
        n_components = int(math.ceil(num_feats_rel*xx.shape[1]))

        match extract:
            case 1:   # ZCA +RBM
                # ZCA
                x = ZCA(xx)
                tx = ZCA(txx)

                # # The `Client` in Builder Pattern for building RTRBM.
                # rt_rbm = RNNRBM(
                #     # The number of units in visible layer.
                #     visible_num=observed_arr.shape[2],
                #     # The number of units in hidden layer.
                #     hidden_num=100,
                #     # The activation function in visible layer.
                #     visible_activating_function=LogisticFunction(),
                #     # The activation function in hidden layer.
                #     hidden_activating_function=LogisticFunction(),
                #     # The activation function in RNN layer.
                #     rnn_activating_function=LogisticFunction(),
                #     # is-a `OptParams`.
                #     opt_params=SGD(),
                #     # Learning rate.
                #     learning_rate=1e-05
                # )

                # # Learning.
                # rt_rbm.learn(
                #     # The `np.ndarray` of observed data points.
                #     observed_arr,
                #     # Training count.
                #     training_count=1000,
                #     # Batch size.
                #     batch_size=20
                # )

                # # error
                # error_arr = rt_rbm.rbm.get_reconstruct_error_arr()

                # plt.figure(figsize=(20, 10))
                # plt.plot(error_arr)
                # plt.ylabel("MSE")
                # plt.xlabel("Epochs")
                # plt.show()

            case 2:   # tSNE

                # Dimensionality reduction to improve performance
                if xx.shape[1] > 10:
                    xx = PCA(n_components=10).fit_transform(xx)
                if txx.shape[1] > 10:
                    txx = PCA(n_components=10).fit_transform(txx)

                # n_components = number of output dimensions (usually 2 for 2D visualization)
                # perplexity = a hyperparameter that controls the balance between preserving global and local structure
                print('\n tSNE extract_features 1')
                tsne = TSNE(n_components=2, perplexity=30, random_state=0)
                print('\n tSNE extract_features 2')
                x = tsne.fit_transform(xx)
                print('\n tSNE extract_features 3')
                tx = tsne.fit_transform(txx)
                print('\n tSNE extract_features 4')

            case 3:   # PCA
                x = PCA(n_components=10).fit_transform(xx)
                tx = PCA(n_components=10).fit_transform(txx)

            case 4:   # ICA
                x = FastICA(n_components=10, whiten='unit-variance').fit_transform(xx)
                tx = FastICA(n_components=10, whiten='unit-variance').fit_transform(txx)

            case 5:   # NMF
                # NMF is a non-convex optimization problem, so the results may vary with different initializations
                x = NMF(n_components=10, init='random', random_state=0).fit_transform(xx)
                tx = NMF(n_components=10, init='random', random_state=0).fit_transform(txx)
                # xx_features = NMF(n_components=10, init='random', random_state=0).components_

            case 6:   # Autoencoder
                shape = xx.shape[1:]
                latent_dim = int(math.ceil(num_feats_rel*shape[0]))
                autoencoder = Autoencoder(latent_dim, shape)
                autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
                autoencoder.fit(xx, xx,
                                epochs=10,
                                shuffle=True,
                                validation_data=(txx, txx))

                x = autoencoder.encoder(xx).numpy()
                tx = autoencoder.encoder(txx).numpy()

            case _:
                x = xx
                tx = txx

        return x, tx

    ######################################################################################
    # FEATURE ENGINEERING
    ######################################################################################
    def engineer_features(x,tx,order=order):
        # print('select_features')
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

        # select features
        if projections:
            matrix=np.random.uniform(0,1,(x.shape[1],num_feats))
            x=np.matmul(x,matrix)
            tx=np.matmul(tx,matrix)
            scaler=MinMaxScaler()
            x=scaler.fit_transform(x)
            tx=scaler.transform(tx)
            return x,tx
        else:
            count=num_feats
            # print('num_feats {}; x shape {}; tx shape {}'.format(num_feats, x.shape, tx.shape))

            if count>x.shape[1]:
                count=x.shape[1]
            feats=np.random.choice(range(x.shape[1]),count,replace=False)
            return x[:,feats],tx[:,feats]

    # no_of_covariates=int(np.ceil(x.shape[1]/10))#3


    ######################################################################################
    # ENSEMBLE
    ######################################################################################
    def one_model(xx,txx):
        # print('one_model')

        x, tx = pre_process(xx,txx)
        x, tx = extract_features(x,tx)
        x, tx = engineer_features(x,tx)

        # eqn. 2 from the DEAN paper
        goal=np.ones(len(x))
        if justlin:
            cv=LinearRegression(fit_intercept=cept).fit(x,goal)
        else:
            cv=LassoCV(fit_intercept=False).fit(x,goal)

        # eqn. 4 from the DEAN paper
        meanv=np.mean(cv.predict(x))
        pred=np.square(cv.predict(tx)-meanv)

        return pred

    scores=[]
    # for i in tqdm(range(no_submodels)):
    for i in range(no_submodels):
        # print('sub model {} of {}'.format(i,no_submodels))
        # print('Selected features of X : {}'.format(np.array(xx).shape))
        pred=one_model(x,tx)
        scores.append(pred)

    #features=list(generate_subsets(list(range(x.shape[1])),num_feats))

    scores=np.array(scores)

    # print('Mean = {}, Variance = {}'.format(np.mean(scores,axis=0),np.var(scores,axis=0)))

    # eqn. 5 from the DEAN paper
    return np.mean(scores,axis=0)

# executes only when ran directly, not when this file is imported into another python file
if __name__ == '__main__':
    sean()