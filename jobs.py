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


with open("configs.json","r") as f:
    configs=json.loads(f.read())

runtime, extract, auc, submodels, altnorm = [], [], [], [], []
for config in configs:
    start_time = time.time()
    pred = sean(X_train, X_test, altnorm=True, no_submodels = config["submodel"], prep=1, extract=config["extract"])
    end_time = time.time()
    rt = end_time - start_time
    runtime.append(rt)
    extract.append(config["extract"])
    submodels.append(config["submodel"])
    altnorm.append(config["altnorm"])
    t_auc = roc_auc_score(y_test, pred)
    auc.append(t_auc)
    print(f'Feature extraction type {config["extract"]} with {config["submodel"]} submodels completed at \
             {datetime.datetime.fromtimestamp(start_time / 1000).strftime("%H:%M:%S")}. AUC: {t_auc}. Runtime: {rt}')
# print('\nROC_AUC: {} '.format(roc_auc_score(y_test, pred)))

# with open('output_data.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Date', 'Dataset', 'Altnorm', 'Feature Extraction', 'Submodel Count', 'Runtime', 'ROC-AUC']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     # Write header if the file is empty
#     if csvfile.tell() == 0:
#         writer.writeheader()

df = pd.DataFrame(
        {
            'Date': datetime.date.today(),
            'Dataset': 'CC Fraud', 
            'Altnorm': altnorm, 
            'Feature Extraction': extract,  
            'Submodel Count':submodels,
            'Runtime': runtime, 
            'ROC-AUC': auc
        }
    )

df.to_csv('output_data.csv')
