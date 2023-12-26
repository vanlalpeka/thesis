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
from tensorflow.keras.datasets import mnist

import csv 
import time 
import json
import os
import datetime

#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename=f"./logs1222/mnist_ae_lin_{datetime.datetime.today()}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

print("START")
logger.info('START')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

try:
    with open("params/p_mnist_ae_lin.csv") as f:
        # heading = next(f) 
        params = csv.DictReader(f, delimiter=';')
        # params=csv.reader(f)

        for param in params:
            for c in range(10):
                for i in range(10):
                    print(f'MNIST ae-lin Class-{c}: {i} out of 10')
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

                    # Print the shapes of the datasets
                    # print(f'X_test : {X_test.shape} {type(X_test)}, Y_test : {Y_test.shape} {type(Y_test)}')
                    pred, ensembles_executed = sean(X_train, 
                                                    X_test, 
                                                    no_submodels = int(param["no_submodels"]), 
                                                    prep=param["prep"], 
                                                    extract=param["extract"], 
                                                    submodel=param["submodel"], 
                                                    interaction_terms_then_randomize=param["interaction_terms_then_randomize"]
                                                    )

                    end_time = time.time()
                    runtime = end_time - start_time
                    auc = roc_auc_score(Y_test, pred)
                    logger.info(f'MNIST Class-{normal_class} \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc} \t {param["interaction_terms_then_randomize"]}')

except Exception:
    logger.exception("message")

logger.info('END')
