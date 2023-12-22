import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDOneClassSVM

from pre_process import *
from feature_selection import *
from feature_bagging import *


def one_model(X_train, X_test, submodel, num_feats_rel, extract, order, prep):
    # print(f'one_model: X_train.shape: {X_train.shape} {submodel}')
    X_train, X_test = pre_process(X_train, X_test, prep)
    X_train, X_test = feature_selection(X_train, X_test, num_feats_rel, extract)
    X_train, X_test = feature_bagging(X_train, X_test, order, num_feats_rel)
    # eqn. 2 from the DEAN paper
    goal=np.ones(len(X_train))

    if submodel ==  "lin":
        cv = LinearRegression(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "lasso":
        cv = LassoCV(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "elastic":
        cv = ElasticNetCV(fit_intercept=False).fit(X_train, goal)
    if submodel ==  "svm":
        cv = SGDOneClassSVM(fit_intercept=False).fit(X_train, goal)

    # eqn. 4 from the DEAN paper
    meanv = np.mean(cv.predict(X_train))
    # print("meanv=np.mean(cv.predict(x)) : ", meanv)
    pred = np.square(cv.predict(X_test)-meanv)
    # print("\n pred=np.square(cv.predict(tx)-meanv) : ", pred)

    return pred