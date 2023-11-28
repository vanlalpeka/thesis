from sean import *

# import importlib.util

# if importlib.util.find_spec('openml') is None:
#     pip install openml
# else:
#     print('openml is already installed')

import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import csv 
import time 
import json
import os
import datetime

#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename=f"./logs/std_{datetime.datetime.today()}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

logger.info('START')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

try:
    with open("params.csv") as f:
        # heading = next(f) 
        params = csv.DictReader(f, delimiter=';')
        # params=csv.reader(f)

        for param in params:
            for i in range(10):
                for c in range(10):
                    start_time = time.time()

                    normal_class = c
                    # Train data
                    xx = x_train[np.isin(y_train, [normal_class]).flatten()]
                    # print("xx set shape:", xx.shape)

                    # Test data: Normal
                    txx = x_test[np.isin(y_test, [normal_class]).flatten()]
                    tyy = np.zeros((len(txx), 1), dtype=int)
                    # print("txx tyy set shape:", txx.shape, tyy.shape)

                    # Test data: Anomalies
                    idx = np.arange(len(tyy))
                    np.random.shuffle(idx)
                    anomalies_count = int(.50*len(idx))     # 50% anomalies

                    anomalies_x_test = x_test[np.isin(y_test, [normal_class], invert=True).flatten()][:anomalies_count]  # "invert=True" get the anomalies i.e. the not normal_class
                    anomalies_y_test = np.ones((anomalies_count, 1), dtype=int)
                    # print("subset_x_test subset_y_test set shape:", anomalies_x_test.shape, anomalies_y_test.shape)

                    txx = np.concatenate((txx, anomalies_x_test))
                    tyy = np.concatenate((tyy, anomalies_y_test))

                    # Print the shapes of the datasets
                    # print("txx tyy set shape:", txx.shape, tyy.shape)

                    pred = sean(xx, txx, no_submodels = int(param["no_submodels"]), prep=param["prep"], extract=param["extract"], submodel=param["submodel"])

                    end_time = time.time()
                    runtime = end_time - start_time
                    auc = roc_auc_score(tyy, pred)
                    logger.info(f'CIFAR10 Class-{normal_class} \t {param["no_submodels"]} \t {param["submodel"]} \t {param["prep"]} \t {param["extract"]} \t {runtime} \t {auc}')

except Exception:
    logger.exception("message")

logger.info('END')
