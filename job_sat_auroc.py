from sean import *

import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import csv 
import time 
import json
import os
import datetime
import sys
#importing the module 
import logging 


# logging.basicConfig(filename=f"./log/cifar10_{datetime.datetime.today()}.log", 
logging.basicConfig(filename=f"./log/sat_auroc.log", 
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
    dataset_id= 40900,  # Satellite
    download_data=True,
    download_qualities=True,
    download_features_meta_data=True,
    )


try:
    for k in range(100,3000,100):
        for i in range(10):
            print(f'Satellite- {i} out of 10')
            start_time = time.time()

            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

            y = y.map({'Normal':0, 'Anomaly':1})
            
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
                        no_submodels = k, 
                        prep='norm', 
                        extract='none', 
                        submodel='lasso', 
                        )

            end_time = time.time()
            runtime = end_time - start_time

            print(f'Job y_test : {np.histogram(y_test, bins=10)}')
            print(f'Job pred: {np.histogram(pred, bins=10)}')

            # for i,j in zip(y_test, pred):
            #     print(i,type(i),j,type(j))
            #     # if int(i) != j:
            #     #     print(i,j)

            auc = roc_auc_score(y_test, pred)
            print(f'AUROC : {auc}')
            logger.info(f'Satellite \t norm \t none \t lasso \t {ensembles_executed} \t {runtime} \t {auc}')

except Exception:
    logger.exception("message")

logger.info('END')
