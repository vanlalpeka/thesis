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

import csv 
import time 
import json
import os
import datetime

#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename=f"std_{datetime.datetime.today()}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 


# Replace dataset_id with the   ID of the dataset you want to load
dataset = openml.datasets.get_dataset(
    dataset_id= 42175,  # CreditCardFraudDetection
    # dataset_id = 40900  # Satellite soil category
    download_data=True,
    download_qualities=True,
    download_features_meta_data=True,
    )

X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Split the data into training and testing sets
# test_size = 0.2  # Adjust the test size as needed (e.g., 0.2 for an 80-20 split)
# random_state = 42  # You can set a random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

logger.info('START')

with open("configs.json","r") as f:
    configs=json.loads(f.read())

# runtime, extract, auc, no_submodels, submodel, altnorm = [], [], [], [], [], []
for config in configs:
    logger.info(f'Calling sean(altnorm={config["altnorm"]}, no_submodels={config["no_submodels"]}, submodel={config["submodel"]}, prep=[], extract={config["extract"]})')
    start_time = time.time()
    pred = sean(X_train, X_test, altnorm=config["altnorm"], no_submodels = config["no_submodels"], prep=[], extract=config["extract"])
    end_time = time.time()
    runtime = end_time - start_time
    auc = roc_auc_score(y_test, pred)
    logger.info(f'Result:: AUC: {auc} Runtime: {runtime}')
# print('\nROC_AUC: {} '.format(roc_auc_score(y_test, pred)))

# with open('output_data.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Date', 'Dataset', 'Altnorm', 'Feature Extraction', 'Submodel Count', 'Runtime', 'ROC-AUC']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     # Write header if the file is empty
#     if csvfile.tell() == 0:
#         writer.writeheader()

# df = pd.DataFrame(
#         {
#             'Date': datetime.date.today(),
#             'Dataset': 'CC Fraud', 
#             'Altnorm': altnorm, 
#             'Feature Extraction': extract,  
#             'Submodel Count':submodels,
#             'Runtime': runtime, 
#             'ROC-AUC': auc
#         }
#     )

# df.to_csv('output_data.csv')
logger.info('END')
