import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import cifar10, mnist

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

logging.basicConfig(filename=f"./logs1222/competitors_{datetime.datetime.today()}.log", 
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
    'Cluster-Based Local Outlier Factor (CBLOF)': CBLOF(n_clusters=10)
}

def compare_classifiers_on_tab_data(ds_id, ds_name):
    dataset = openml.datasets.get_dataset(
        # dataset_id= 42175,  # CreditCardFraudDetection
        dataset_id= ds_id,  # CreditCardFraudDetection
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
        )

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if ds_name == "Satellite":
        y = y.map({'Normal':0, 'Anomaly':1})

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    for _, (clf_name, clf) in enumerate(classifiers.items()):

        for i in range(10):
            print(f'{i} out of 10')

            start_time = time.time()

            # Identify indices of samples where y=1 (fraudulent transactions)
            fraud_indices = [i for i, label in enumerate(y) if label == 1]

            X_fraud = X[fraud_indices]
            y_fraud = y[fraud_indices]

            X_no_fraud = np.delete(X, fraud_indices, axis=0)
            y_no_fraud = np.delete(y, fraud_indices, axis=0)

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_no_fraud, y_no_fraud, test_size=0.2)

            # Include all samples with y=1 in the test set
            X_test = np.concatenate((X_test, X_fraud), axis=0)
            y_test = np.concatenate((y_test, y_fraud), axis=0)

            clf.fit(X_train)

            y_test_scores = clf.decision_function(X_test)  # outlier scores

            end_time = time.time()
            runtime = end_time - start_time
            auc = roc_auc_score(y_test, y_test_scores)

            # print(f' y_test.shape {y_test.shape} ; pred.shape {pred.shape}')
            print(f'{ds_name} {clf_name} {runtime} sec., AUROC = {auc}')
            logger.info(f'{ds_name} {clf_name} {runtime} {auc}')


def compare_classifiers_on_img_data(ds_name):
    if ds_name == 'CIFAR-10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if ds_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for _, (clf_name, clf) in enumerate(classifiers.items()):
        for c in range(10):  # normal class
            for i in range(10):
                print(f'Class-{c}: {i} out of 10')

                start_time = time.time()

                normal_class = c
                # Train data
                X_train = x_train[np.isin(y_train, [normal_class])]
                # Y_train = y_train[np.isin(y_train, [normal_class])]
                # print("X_train.shape:", X_train.shape)

                # Test data: Normal
                X_test = x_test[np.isin(y_test, [normal_class])]
                Y_test = np.zeros((len(X_test), 1), dtype=int)
                # print("X_test Y_test set shape:", X_test.shape, Y_test.shape)

                # Test data: Anomalies
                idx = np.arange(len(Y_test))
                np.random.shuffle(idx)
                anomalies_count = int(.50*len(idx))

                anomalies_X_test = x_test[np.isin(y_test, [normal_class], invert=True)][:anomalies_count]  # "invert=True" get the anomalies i.e. the not normal_class
                anomalies_Y_test = np.ones((anomalies_count, 1), dtype=int)
                # print("subset_x_test subset_y_test set shape:", anomalies_X_test.shape, anomalies_Y_test.shape)

                X_test = np.concatenate((X_test, anomalies_X_test))
                Y_test = np.concatenate((Y_test, anomalies_Y_test))

                # X_train = X_train.reshape(X_train.shape[0], -1)
                # X_test = X_test.reshape(X_test.shape[0], -1)

                clf.fit(X_train)
                Y_test_scores = clf.decision_function(X_test)  # outlier scores

                end_time = time.time()
                runtime = end_time - start_time
                auc = roc_auc_score(Y_test, Y_test_scores)

                print(f'{ds_name} {clf_name}: Normal class = {normal_class}, {runtime} sec., AUROC = {auc}')
                logger.info(f'{ds_name} {clf_name} Class-{normal_class} {runtime} {auc}')


compare_classifiers_on_tab_data(42175, "CCFraud")  # CreditCardFraudDetection
compare_classifiers_on_tab_data(40900, "Satellite")  # Satellite soil category
compare_classifiers_on_img_data("CIFAR-10")  # CIFAR-10
compare_classifiers_on_img_data("MNIST")  # MNIST