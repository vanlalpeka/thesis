import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf

from pyod.models.knn import KNN   # kNN detector
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF

from pyod.utils.example import visualize
from tqdm import tqdm

import time 
import json
import os
import datetime
import logging 

logging.basicConfig(filename=f"./logs/comps_ccfraud_{datetime.datetime.today()}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 
logger.info('START')
print("START")

classifiers = {
    'Isolation Forest':IForest(),
    'K Nearest Neighbors (KNN)': KNN(),
    # 'Angle-based Outlier Detector (ABOD)': ABOD(),
    'Cluster-Based Local Outlier Factor (CBLOF)': CBLOF()
}

dataset = openml.datasets.get_dataset(
    dataset_id= 42175,  # CreditCardFraudDetection
    download_data=True,
    download_qualities=True,
    download_features_meta_data=True,
    )

for _, (clf_name, clf) in enumerate(classifiers.items()):
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    for i in range(10):
        print(f'{i} out of 10')

        start_time = time.time()

        # Identify indices of samples where y=1 (fraudulent transactions)
        fraud_indices = [i for i, label in enumerate(y) if label == 1]

        X_fraud = X.loc[fraud_indices]
        y_fraud = y.loc[fraud_indices]

        X_no_fraud = X.drop(fraud_indices)
        y_no_fraud = y.drop(fraud_indices)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_no_fraud, y_no_fraud, test_size=0.2)

        # Include all samples with y=1 in the test set
        X_test = pd.concat([X_test, X_fraud], axis=0)
        y_test = pd.concat([y_test, y_fraud], axis=0)

        clf.fit(X_train)

        # # get the prediction label and outlier scores of the training data
        # Y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        # Y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        # y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        # # it is possible to get the prediction confidence as well
        # y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

        end_time = time.time()
        runtime = end_time - start_time
        # auc = roc_auc_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_scores)

        # print(f' y_test.shape {y_test.shape} ; pred.shape {pred.shape}')
        print(f'\n CCFraud {clf_name} ROC_AUC = {auc}')
        logger.info(f' {clf_name} CCFraud {runtime} \t {auc}')

        # # visualize the results
        # visualize(clf_name, X_train, Y_train, X_test, y_test, Y_train_pred, y_test_pred, show_figure=True, save_figure=False)