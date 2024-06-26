from sean import *

# Use unverified SSL to avoid SSL Certificate error 1006 in LiDO3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.datasets import cifar10

import csv 
import time 
import json
import os
import datetime
import sys
import logging 

k = 1
row = int(sys.argv[1])

# logging.basicConfig(filename=f"./log/cifar10_{datetime.datetime.today()}.log", 
logging.basicConfig(filename=f"./log/cifar10_{row}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 
logger.info('START')

print("START")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

try:
    with open("params/img.csv") as f:
        params = csv.DictReader(f, delimiter=';')
        for temp in params:
            if k == row:
                param = temp
            k = k+1

        for c in range(10): # 10 classes
            for i in range(10): # Run 10 times for each class
                print(f'CIFAR-10 Class-{c}: {i} out of 10. {param}')
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
                anomalies_count = len(idx)

                anomalies_X_test = x_test[np.isin(y_test, [normal_class], invert=True).flatten()][:anomalies_count]  # "invert=True" get the anomalies i.e. the not normal_class
                anomalies_Y_test = np.ones((anomalies_count, 1), dtype=int)
                # print("subset_x_test subset_y_test set shape:", anomalies_X_test.shape, anomalies_Y_test.shape)

                X_test = np.concatenate((X_test, anomalies_X_test))
                Y_test = np.concatenate((Y_test, anomalies_Y_test))

                # Print the shapes of the datasets
                # print(f'X_test : {X_test.shape} {type(X_test)}, Y_test : {Y_test.shape} {type(Y_test)}')

                pred, ensembles_executed = sean(X_train, 
                                                X_test, 
                                                no_submodels = int(param["no_submodels"]), 
                                                prep=param["prep"].split(','), 
                                                extract=param["extract"], 
                                                submodel=param["submodel"], 
                                                )

                end_time = time.time()
                runtime = end_time - start_time
                auc = roc_auc_score(Y_test, pred)
                logger.info(f'CIFAR-10 Class-{normal_class} \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc}')

except Exception:
    logger.exception("message")

logger.info('END')
