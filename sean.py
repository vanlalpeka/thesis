import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from pre_process import *
from feature_selection import *
from feature_bagging import *
from one_model import *


def sean(X_train, X_test, no_submodels=5000, num_feats_rel=0.2, order=2, prep=[], extract='ica', submodel='lin',feat_reduce_then_bagging=True, interaction_terms_then_randomize=True, desired_variance_ratio = 0.95):

    # Set the maximum computation time (in seconds)
    max_computation_time = 600  # 10 minutes
    start_time = time.time()

    # X_train, X_test = pre_process(X_train, X_test, prep)

    # if feat_reduce_then_bagging:
    #     X_train, X_test = feature_selection(X_train, X_test, num_feats_rel, extract)
    #     X_train, X_test = feature_bagging(X_train, X_test, order, num_feats_rel)
    # else:
    #     X_train, X_test = feature_bagging(X_train, X_test, order, num_feats_rel)
    #     X_train, X_test = feature_selection(X_train, X_test, num_feats_rel, extract)

    scores=[]
    ensembles_executed = 0

    elapsed_time = time.time() - start_time
    if elapsed_time > max_computation_time:
        print("Feature reduction: Time limit reached. Exiting.")
        return np.zeros(X_test.shape[0]), ensembles_executed

    else:
        for i in tqdm(range(no_submodels)):
            # pred = one_model(X_train, X_test, submodel)
            pred = one_model(X_train.to_numpy(), X_test.to_numpy(), submodel, num_feats_rel, extract, order, prep)
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