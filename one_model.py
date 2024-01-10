##############################################################################################################
# This is the submodel of the ensemble
# It runs, in sequence, the feature_bagging() and a simple submodel.
##############################################################################################################

import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDOneClassSVM
# from scipy.interpolate import CubicSpline
# from pyearth import Earth

from pre_process import *
from feature_selection import *
from feature_bagging import *


def one_model(X_train, X_test, submodel, feat_sel_percent, extract, order, prep):
    # print(f'one_model: X_train.shape: {X_train.shape} {submodel}')
    # X_train, X_test = pre_process(X_train, X_test, prep)
    # X_train, X_test = feature_selection(X_train, X_test, feat_sel_percent, extract)
    X_train, X_test = feature_bagging(X_train, X_test, order, feat_sel_percent)
    # eqn. 2 from the DEAN paper
    goal = np.ones(len(X_train))

    # if submodel == "spline":
    #     cs = CubicSpline(x, y, bc_type='natural')
    #     # eqn. 4 from the DEAN paper
    #     meanv = np.mean(cs(X_train))
    #     pred = np.square(cs(X_test)-meanv)
    # else:

    if submodel ==  "lin":
        cv = LinearRegression(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "lasso":
        cv = LassoCV(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "elastic":
        cv = ElasticNetCV(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "svm":
        cv = SGDOneClassSVM(fit_intercept=False).fit(X_train, goal)
    # if submodel ==  "mars":
    #     cv = Earth().fit(X_train, goal)

    # eqn. 4 from the DEAN paper
    meanv = np.mean(cv.predict(X_train))
    pred = np.square(cv.predict(X_test) - meanv)

    return pred