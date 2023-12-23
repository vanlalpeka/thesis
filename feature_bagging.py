# !pip install openml

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

import math
import itertools

######################################################################################
# FEATURE BAGGING
# First, standardize (mean=0, variance=1) the data; second, generate interaction terms.
# This is to prevent multicollinearity among the new (engineered) features
# https://www.tandfonline.com/doi/abs/10.1080/01621459.1980.10477430
######################################################################################
def feature_bagging(X_train, X_test, order, num_feats_rel):
    # print(f'feature_bagging num_feats_rel: {num_feats_rel}')
    # generate different set of combination of features
    # the count of features in a feature set is bounded by the order value

    # #all ordered subsequences up to count of length up to order
    # orderings = list(itertools.chain.from_iterable(itertools.combinations(range(X_train.shape[1]),i) for i in range(1,order+1)))
    # print(orderings)

    # num_feats = int(math.ceil(num_feats_rel*len(orderings)))

    # # update order
    # xx,txx=[],[]
    # for order in orderings:
    #     # print('update_order for-loop: order =={}'.format(order))
    #     xx.append(np.prod([X_train[:,i] for i in order],axis=0))
    #     txx.append(np.prod([X_test[:,i] for i in order],axis=0))
    # # print('update order: xx.shape {}'.format(np.array(xx).shape))

    # x = np.array(xx).T
    # tx = np.array(txx).T

    # if num_feats > x.shape[1]:
    #     num_feats = x.shape[1]
    # print(num_feats)
    # feats = np.random.choice(range(x.shape[1]), num_feats, replace=False)
    # print(feats)

    # return x[:,feats],tx[:,feats]


    # Build interaction terms using PolynomialFeatures
    poly = PolynomialFeatures(degree=order, include_bias=False, interaction_only=True)
    X_train_interaction_terms = poly.fit_transform(X_train)
    X_test_interaction_terms = poly.transform(X_test)

    num_feats = int(math.ceil(num_feats_rel*X_train_interaction_terms.shape[1]))

    # Select a random subset of features (excluding the interaction terms)
    selected_features = np.random.choice(range(X_train_interaction_terms.shape[1]), size=num_feats, replace=False)
    # print('interaction_terms_then_randomize : selected_features ', selected_features)

    # Extract the selected features from the interaction terms
    X_train = X_train_interaction_terms[:,selected_features]
    X_test = X_test_interaction_terms[:,selected_features]
    # print(X_train.shape)

    return X_train, X_test
