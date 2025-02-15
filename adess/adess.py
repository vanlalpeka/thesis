#######################################################################################################################
# Say, the ensemble size is 50.
# This function will call pre_process() and feature_selection() once each, 
# and then call one_model() 50 times.
#
# one_model() handles both feature bagging and the sub-model.
#######################################################################################################################

import numpy as np
import time 
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import argparse

from adess.pre_process import pre_process
from adess.feature_selection import feature_selection
from adess.one_model import one_model

# A custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

def print_result(pred, ensembles_executed):
    print(f'Mean of Predicted Y = {np.mean(np.array(pred, np.float64))}, Count of submodel executed = {ensembles_executed}')
    
def adess(X_train, X_test, no_submodels=500, feat_sel_percent=0.2, max_feats = 50, order=2, prep=['norm'], extract='pca', submodel='lin', computation_budget=600):
    # print("adess.py: adess()")
    """
    Description:
    An ensemble of submodels.
    
    Parameters:
    X_train (ndarray): Training data
    X_test (ndarray): Testing data
    feat_sel_percent (float): Feature selection percentage, default=0.2
    max_feats (int): Maximum number of features, default=50
    order (int): Degree of polynomials for feature bagging, default=2
    computation_budget (int): Computation budget in seconds, default=600
    no_submodels (int): Count of submodels in the ensemble, default=5000
    prep (list): A list of pre-processing methods. Options (choose one or many) [skel,canny,clahe,blur,augment,gray,norm,std,none], default='norm'
    extract (str): A feature selection method. Options (choose one) [rbm,tsne,pca,ica,nmf,ae,none], default='ae'
    submodel (str): A submodel for the ensemble. Options (choose one) [lin,lasso,elastic,svm], default='lin'

    Returns:
    float: Predicted Y
    int: Count of submodel executed
    """

    # Computation budget (in seconds)
    computation_budget = computation_budget  # 10 minutes
    start_time = time.time()

    X_train, X_test = pre_process(X_train, X_test, prep)
    X_train, X_test = feature_selection(X_train, X_test, feat_sel_percent, max_feats, extract)
    # print(f'After feature selection: X_train.shape {X_train.shape}, X_test.shape {X_test.shape}')

    scores=[]
    count_of_submodels_executed = 0

    elapsed_time = time.time() - start_time

    if elapsed_time > computation_budget:
        return np.zeros(X_test.shape[0]), count_of_submodels_executed

    else:    
        # Feature bagging
        # Build interaction terms using PolynomialFeatures. interaction_only=True.
        poly = PolynomialFeatures(degree = order, include_bias=False, interaction_only=True)

        X_train_interaction_terms = poly.fit_transform(X_train)
        X_test_interaction_terms = poly.transform(X_test)
        # print(f'After PolynomialFeatures: X_train_interaction_terms.shape {X_train_interaction_terms.shape}, X_test_interaction_terms.shape {X_test_interaction_terms.shape}')

        for _ in tqdm(range(no_submodels)):
            
            pred = one_model(X_train_interaction_terms, X_test_interaction_terms, feat_sel_percent, max_feats, submodel)
            # pred = one_model(X_train, X_test, feat_sel_percent,  order, prep, extract, submodel)
            scores.append(pred)

            count_of_submodels_executed += 1

            elapsed_time = time.time() - start_time
            if elapsed_time > computation_budget:
                break

        # scores=np.array(scores)

        # eqn. 5 from the DEAN paper
        return np.mean(np.array(scores, np.float64)), count_of_submodels_executed


# executes only when run directly, not when this file is imported into another python file
if __name__ == '__main__':
    X, y= load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pred, ensembles_executed = adess(X_train, X_test)

    print(f'X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}, Mean of Predicted Y = {pred}, Count of submodel executed = {ensembles_executed}')
