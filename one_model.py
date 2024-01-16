import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDOneClassSVM
# from scipy.interpolate import CubicSpline
# from pyearth import Earth

from pre_process import *
from feature_selection import *
from feature_bagging import *


def one_model(X_train, X_test, feat_sel_percent,  order, prep, extract, submodel):
    """
    This is the submodel of the ensemble.
    It runs the feature_bagging(), followed by a simple submodel.

    X_train and X_test are ndarray of the train and the test sets.

    feat_sel_percent: The percentage of features to select e.g. 0.2 means select 20% of the original features.

    order: Degree of polynomials for feature bagging.

    prep: A list of pre-processing methods. It can be an empty list, in which case no preprocessing will be done; 
    except for image file, which will be flattend regardless of this field

    extract: A feature selection method. Options are ica, pca, nmf, rbm, ae, tsne.
    
    submodel: A submodel for the ensemble. Options are lin, svm, lasso, elastic.
    
    """
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