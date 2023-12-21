import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.datasets import cifar10   #, mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
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

logging.basicConfig(filename=f"./logs/comps_cifar10_{datetime.datetime.today()}.log", 
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
    'Cluster-Based Local Outlier Factor (CBLOF)': CBLOF(random_state=42, n_clusters=10)
}

for _, (clf_name, clf) in enumerate(classifiers.items()):
    # Load CIFAR-10 data
    # CIFAR-10 contains 6000 images per class.
    # The original train-test split randomly divided these into 5000 train and 1000 test images per class.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for c in range(10):
        for i in range(10):
            print(f'Class-{c}: {i} out of 10')

            start_time = time.time()

            normal_class = c
            # Train data
            X_train = x_train[np.isin(y_train, [normal_class]).flatten()]
            # Y_train = y_train[np.isin(y_train, [normal_class]).flatten()]
            # print("X_train.shape:", X_train.shape)

            # Test data: Normal
            X_test = x_test[np.isin(y_test, [normal_class]).flatten()]
            Y_test = np.zeros((len(X_test), 1), dtype=int)
            # print("X_test Y_test set shape:", X_test.shape, Y_test.shape)

            # Test data: Anomalies
            idx = np.arange(len(Y_test))
            np.random.shuffle(idx)
            anomalies_count = int(.50*len(idx))

            anomalies_X_test = x_test[np.isin(y_test, [normal_class], invert=True).flatten()][:anomalies_count]  # "invert=True" get the anomalies i.e. the not normal_class
            anomalies_Y_test = np.ones((anomalies_count, 1), dtype=int)
            # print("subset_x_test subset_y_test set shape:", anomalies_X_test.shape, anomalies_Y_test.shape)

            X_test = np.concatenate((X_test, anomalies_X_test))
            Y_test = np.concatenate((Y_test, anomalies_Y_test))

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            # Print the shapes of the datasets
            # print("X_test Y_test set shape:", X_test.shape, Y_test.shape)

            # Y_test_pred = sean(X_train, X_test, no_submodels = 5, prep=["std", "canny", "clahe", "blur", "augment","gray"], extract="pca", submodel='ridge', feat_reduce_then_bagging=True, interaction_terms_then_randomize=True)

            clf.fit(X_train)

            # # get the prediction label and outlier scores of the training data
            # Y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            # Y_train_scores = clf.decision_scores_  # raw outlier scores

            # get the prediction on the test data
            # Y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
            Y_test_scores = clf.decision_function(X_test)  # outlier scores

            # # it is possible to get the prediction confidence as well
            # Y_test_pred, Y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

            end_time = time.time()
            runtime = end_time - start_time
            # auc = roc_auc_score(Y_test, Y_test_pred)
            auc = roc_auc_score(Y_test, Y_test_scores)

            # print(f' Y_test.shape {Y_test.shape} ; pred.shape {pred.shape}')
            print(f'\n CIFAR10 {clf_name}: Normal class = {normal_class}, ROC_AUC = {auc}')
            logger.info(f' {clf_name} CIFAR10 Class-{normal_class} \t {runtime} \t {auc}')

            # # visualize the results
            # visualize(clf_name, X_train, Y_train, X_test, Y_test, Y_train_pred, Y_test_pred, show_figure=True, save_figure=False)