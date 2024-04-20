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
# from tensorflow.keras.datasets import cifar10

import csv 
import time 
import json
import os
import datetime
import sys
#importing the module 
import logging 

k = 1
row = int(sys.argv[1])

# logging.basicConfig(filename=f"./log/cifar10_{datetime.datetime.today()}.log", 
logging.basicConfig(filename=f"./log/ccfraud_{row}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

print("START")
logger.info('START')

# Replace dataset_id with the   ID of the dataset you want to load
dataset = openml.datasets.get_dataset(
    dataset_id= 42175,  # CreditCardFraudDetection
    download_data=True,
    download_qualities=True,
    download_features_meta_data=True,
    )


try:
    with open("params/tab.csv") as f:
        params = csv.DictReader(f, delimiter=';')
        for temp in params:
            if k == row:
                param = temp
            k = k+1

        for i in range(10):
            print(f'CCFraud-{param}: {i} out of 10')
            start_time = time.time()

            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

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

            pred, ensembles_executed = sean(X_train.to_numpy(), 
                        X_test.to_numpy(), 
                        no_submodels = int(param["no_submodels"]), 
                        prep=param["prep"], 
                        extract=param["extract"], 
                        submodel=param["submodel"], 
                        )

            end_time = time.time()
            runtime = end_time - start_time
            auc = roc_auc_score(y_test, pred)
            print(f'AUROC : {auc}')
            logger.info(f'CCFraud \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc}')

except Exception:
    logger.exception("message")

logger.info('END')
