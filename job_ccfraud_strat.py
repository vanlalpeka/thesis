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

#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename=f"./logs/ccfraud_100_strat_{datetime.datetime.today()}.log", 
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
    with open("params_ccfraud_100.csv") as f:
        # heading = next(f) 
        params = csv.DictReader(f, delimiter=';')
        # params=csv.reader(f)

        for param in params:
            for i in range(10):
                print(f'CCFraud contaminated train set-{param}: {i} out of 10')
                start_time = time.time()

                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

                # Split the data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

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
                auc = roc_auc_score(y_test, pred)
                logger.info(f'CCFraud \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc} \t {param["interaction_terms_then_randomize"]}')

except Exception:
    logger.exception("message")

logger.info('END')
