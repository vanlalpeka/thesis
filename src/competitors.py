########################################################################################################
# Runs KNN, Isolation Forest, and CBLOF from the pyod package.
# The train:test ratio is 80:20 for the credit card fraud and the satellite datasets, and 2:1 for CIFAR-10 and MNIST
# A classifier is run 10 times each for a given train:test set, and the average of the 10 will be compared with other classifiers.
########################################################################################################
import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import cifar10, mnist

from pyod.models.knn import KNN   # kNN detector
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF

from pyod.utils.example import visualize
from tqdm import tqdm

import time 
import logging 
import os

# create log directory if it does not exist
os.makedirs("log/", exist_ok=True)

logging.basicConfig(filename=f"log/AUROC_competition.log", 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    filemode='w',
                    level=logging.INFO) 

logger=logging.getLogger(__name__) 
logger.info('START')
# print("START")

classifiers = {
    'Isolation Forest':IForest(),
    'K Nearest Neighbors (KNN)': KNN(),
    # 'Angle-based Outlier Detector (ABOD)': ABOD(),
    'Cluster-Based Local Outlier Factor (CBLOF)': CBLOF(n_clusters=10)
}

# This function is for the tabular dataset i.e. credit card and satellite
def compare_classifiers_on_tab_data(ds_id, ds_name):
    dataset = openml.datasets.get_dataset(
        dataset_id= ds_id,  
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
        )

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Satellite dataset needs re-mapping the 'y'
    if ds_name == "Satellite":
        y = y.map({'Normal':0, 'Anomaly':1})

    # Normalize the X prior to classification
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Run the classifiers on the train:test set
    for _, (clf_name, clf) in enumerate(classifiers.items()):
        # Run a classifier 10 times on the train:test set
        for i in range(10):
            # print(f'{i} out of 10')
            start_time = time.time()

            # Train:Test split
            # The train set has no anomaly
            ##################################################################
            # Identify indices of samples where y=1 (anomaly)
            anomaly_indices = [i for i, label in enumerate(y) if label == 1]

            X_anomaly = X[anomaly_indices]
            y_anomaly = y[anomaly_indices]

            X_no_anomaly = np.delete(X, anomaly_indices, axis=0)
            y_no_anomaly = np.delete(y, anomaly_indices, axis=0)

            # Split the set of "no anomalies" into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_no_anomaly, y_no_anomaly, test_size=0.2)

            # Append the anomalies to the test set. 
            X_test = np.concatenate((X_test, X_anomaly), axis=0)
            y_test = np.concatenate((y_test, y_anomaly), axis=0)
            ##################################################################

            clf.fit(X_train)

            y_test_scores = clf.decision_function(X_test)  # outlier scores

            end_time = time.time()
            runtime = end_time - start_time
            auc = roc_auc_score(y_test, y_test_scores)

            # print(f' y_test.shape {y_test.shape} ; pred.shape {pred.shape}')
            # print(f'{ds_name} {clf_name} {runtime} sec., AUROC = {auc}')
            logger.info(f'{ds_name} {clf_name} {runtime} {auc}')


# This function is for the image dataset i.e. CIFAR-10 and MNIST
def compare_classifiers_on_img_data(ds_name):
    if ds_name == 'CIFAR-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if ds_name == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten X
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # print("X_train.shape:", X_train.shape)

    # Normalize X
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Run the classifiers 
    for _, (clf_name, clf) in enumerate(classifiers.items()):
        # In each loop, one class is considered as normal and the other classes as anomalies
        for c in range(10):  # normal class
            # Run a classifier 10 times on the train:test set
            for i in range(10):
                start_time = time.time()

                # Train:Test split  
                ##################################################################
                normal_class = c
                # Train data of the normal class
                XX_train = X_train[np.isin(y_train, [normal_class]).flatten()]
                # print("XX_train.shape:", XX_train.shape)

                # Test data of the normal class
                XX_test = X_test[np.isin(y_test, [normal_class]).flatten()]
                yy_test = np.zeros((len(XX_test), 1), dtype=int)  # y=0 for normal class
                # print("X_test Y_test set shape:", X_test.shape, Y_test.shape)

                # Test data of the anomalies i.e. the not normal classes
                idx = np.arange(len(yy_test))
                np.random.shuffle(idx)
                anomalies_count = int(.50*len(idx))  # The test data has normal:anomaly ratio of 2:1
                anomalies_XX_test = X_test[np.isin(y_test, [normal_class], invert=True).flatten()][:anomalies_count]  # "invert=True" get the anomalies i.e. the not normal_class
                anomalies_yy_test = np.ones((anomalies_count, 1), dtype=int)  # y=1 for the anomalies
                # print("subset_x_test subset_y_test set shape:", anomalies_X_test.shape, anomalies_Y_test.shape)

                XX_test = np.concatenate((XX_test, anomalies_XX_test))
                yy_test = np.concatenate((yy_test, anomalies_yy_test))
                ##################################################################

                clf.fit(XX_train)
                yy_test_scores = clf.decision_function(XX_test)  # outlier scores

                end_time = time.time()
                runtime = end_time - start_time
                auc = roc_auc_score(yy_test, yy_test_scores)

                # print(f'{ds_name} {clf_name} {i} out of 10: Normal class = {normal_class}, {runtime} sec., AUROC = {auc}')
                logger.info(f'{ds_name} {clf_name} Class-{normal_class} {runtime} {auc}')


compare_classifiers_on_tab_data(42175, "CCFraud")  # CreditCardFraudDetection
compare_classifiers_on_tab_data(40900, "Satellite")  # Satellite soil category
compare_classifiers_on_img_data("CIFAR-10")  # CIFAR-10
compare_classifiers_on_img_data("MNIST")  # MNIST
