#######################################################################################################################
# Say, the ensemble size is 50.
# This function will call pre_process() and feature_selection() once each, 
# and then call one_model() 50 times.
#
# one_model() handles both feature bagging and the sub-model.
#######################################################################################################################

import numpy as np
import pandas as pd
import csv 
import time 
import json
import os
import datetime
import sys
import logging 
import openml
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


from pre_process import *
from feature_selection import *
# from feature_bagging import *
from one_model import *

def sean(X_train, X_test, no_submodels=5000, feat_sel_percent=0.2, max_feats = 50, order=2, prep=[], extract='ica', submodel='lin'):
    """
    X_train and X_test are ndarray of the train and the test sets.

    no_submodels: The count of the sub-models for the ensemble.

    feat_sel_percent: The percentage of features to select e.g. 0.2 means select 20% of the original features.

    max_feats: The maximum number of features to select.

    order: Degree of polynomials for feature bagging.

    prep: A list of pre-processing methods. It can be an empty list, in which case no preprocessing will be done; 
    except for image file, which will be flattend regardless of this field

    extract: A feature selection method. Options are ica, pca, nmf, rbm, ae, tsne.
    
    submodel: A submodel for the ensemble. Options are lin, svm, lasso, elastic.
    """

    # Computation budget (in seconds)
    computation_budget = 600  # 10 minutes
    start_time = time.time()

    X_train, X_test = pre_process(X_train, X_test, prep)
    X_train, X_test = feature_selection(X_train, X_test, feat_sel_percent, max_feats, extract)
    # print(f'After feature selection: X_train.shape {X_train.shape}, X_test.shape {X_test.shape}')

    scores=[]
    count_of_submodels_executed = 0

    elapsed_time = time.time() - start_time

    if elapsed_time > computation_budget:
        print("feature_selection(): Time limit reached. Exiting.")
        return np.zeros(X_test.shape[0]), count_of_submodels_executed

    else:    
        poly = PolynomialFeatures(degree = order, include_bias=False, interaction_only=True)

        X_train_interaction_terms = poly.fit_transform(X_train)
        X_test_interaction_terms = poly.transform(X_test)
        # print(f'After PolynomialFeatures: X_train_interaction_terms.shape {X_train_interaction_terms.shape}, X_test_interaction_terms.shape {X_test_interaction_terms.shape}')

        for _ in tqdm(range(no_submodels)):
            
            pred = one_model(X_train_interaction_terms, X_test_interaction_terms, feat_sel_percent, max_feats, order, prep, extract, submodel)
            # pred = one_model(X_train, X_test, feat_sel_percent,  order, prep, extract, submodel)
            scores.append(pred)

            count_of_submodels_executed += 1

            elapsed_time = time.time() - start_time
            if elapsed_time > computation_budget:
                print("Ensemble: Time limit reached. Exiting.")
                break

        scores=np.array(scores)

        # eqn. 5 from the DEAN paper
        return np.mean(scores,axis=0), count_of_submodels_executed


# executes only when run directly, not when this file is imported into another python file
if __name__ == '__main__':
    sean()