import numpy as np
# import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

import math
import itertools

def feature_bagging(X_train, X_test, feat_sel_percent, order):
    """
    X_train and X_test are ndarray of the train and the test sets.
    Bagging with interaction terms only.

    feat_sel_percent: The percentage of features to select e.g. 0.2 means select 20% of the original features.

    order: Degree of polynomials for feature bagging.
    """

    # Build interaction terms using PolynomialFeatures. interaction_only=True.
    poly = PolynomialFeatures(degree = order, include_bias=False, interaction_only=True)

    X_train_interaction_terms = poly.fit_transform(X_train)
    X_test_interaction_terms = poly.transform(X_test)

    num_feats = int(math.ceil(feat_sel_percent*X_train_interaction_terms.shape[1]))

    selected_features = np.random.choice(range(X_train_interaction_terms.shape[1]), size=num_feats, replace=False)
    # print('interaction_terms_then_randomize : selected_features ', selected_features)

    # Extract the selected features from the interaction terms
    X_train = X_train_interaction_terms[:,selected_features]
    X_test = X_test_interaction_terms[:,selected_features]
    # print(X_train.shape)

    return X_train, X_test
