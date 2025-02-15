import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDOneClassSVM
# from scipy.interpolate import CubicSpline
# from pyearth import Earth
import math

def one_model(X_train_interaction_terms, X_test_interaction_terms, feat_sel_percent, max_feats, submodel):
    """
    This is the submodel of the ensemble.

    X_train and X_test are ndarray of the train and the test sets.

    X_train_interaction_terms, X_test_interaction_terms are the interaction terms from X_train and X_test respectively.

    feat_sel_percent: The percentage of features to select e.g. 0.2 means select 20% of the original features.

    max_feats: The maximum number of features to select.
    
    submodel: A submodel for the ensemble. Options are lin, svm, lasso, elastic.
    
    """
    # print(f'one_model: X_train.shape: {X_train.shape} {submodel}')
    

    # Limit the number of features to max_feats
    n_components = int(math.ceil(feat_sel_percent*X_train_interaction_terms.shape[1]))
    if n_components > max_feats:
        n_components = max_feats

    selected_features = np.random.choice(range(X_train_interaction_terms.shape[1]), size=n_components, replace=False)

    # Extract the selected features from the interaction terms
    X_train = X_train_interaction_terms[:,selected_features]
    X_test = X_test_interaction_terms[:,selected_features]

    # print(f'one_model() selected_features: X_train.shape {X_train.shape}, X_test.shape {X_test.shape}')

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
    # print(cv.predict(X_train))
    temp = cv.predict(X_train)
    meanv = np.mean(temp)
    # meanv = np.mean(cv.predict(X_train))
    # if meanv != 1.0:
    # print(meanv)
    # meanv = np.ones(len(X_train))
    pred = np.square(cv.predict(X_test) - meanv)
    # print(f'one_model() cv.predict(X_train) : {np.histogram(temp, bins=10)}')
    # print(f'one_model() meanv : {np.histogram(meanv, bins=10)}')
    # print(f'one_model() cv.predict(X_test) : {np.histogram(cv.predict(X_test), bins=10)}')
    # print(f'one_model() pred : {np.histogram(pred, bins=10)}')

    return pred
    # return np.square(cv.predict(X_test))

if __name__ == '__main__':
    one_model()