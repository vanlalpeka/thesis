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


# Replace dataset_id with the   ID of the dataset you want to load
dataset = openml.datasets.get_dataset(
    dataset_id= 42175,  # CreditCardFraudDetection
    # dataset_id = 40900,  # Satellite soil category
    # dataset_id = 40927, # CIFAR-10
    download_data=True,
    download_qualities=True,
    download_features_meta_data=True,
    )

X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Split the data into training and testing sets
# test_size = 0.2  # Adjust the test size as needed (e.g., 0.2 for an 80-20 split)
# random_state = 42  # You can set a random seed for reproducibility
t_xx, t_txx, t_yy, t_tyy = train_test_split(X, y,test_size=0.2, random_state=42)

# Load CIFAR-10 data
# CIFAR-10 contains 6000 images per class.
# The original train-test split randomly divided these into 5000 train and 1000 test images per class.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# normal class = 0
# 10% anomaly
index = np.where(y_train == 0)
xx = x_train[index]
index = np.where(y_train != 0)
temp = x_train[index]
sample_size = int(0.5*xx.shape[0])
xx=np.append(xx, temp[np.random.choice(len(temp), size = sample_size, replace=False)], axis=0)

index = np.where(y_test == 0)
txx = x_test[index]
tyy = np.zeros(len(txx)) # not anomalies
index = np.where(y_test != 0)
temp1 = x_test[index]
temp2 = np.ones(len(temp1)) # anomalies

sample_size = int(0.5*txx.shape[0])
txx = np.append(txx, temp1[np.random.choice(len(temp1), size = sample_size, replace=False)], axis=0)
tyy = np.append(tyy, temp2[np.random.choice(len(temp2), size = sample_size, replace=False)], axis=0)

i_xx = np.expand_dims(xx, -1)
i_txx = np.expand_dims(txx, -1)
i_tyy = np.expand_dims(tyy, -1)

#now we will Create and configure logger 
logging.basicConfig(filename=f"./logs/std_{datetime.datetime.today()}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

logger.info('START')

# with open("configs.json","r") as f:
#     configs=json.loads(f.read())

try:
    with open("params.csv") as f:
        # heading = next(f) 
        params = csv.DictReader(f, delimiter=';')
        # params=csv.reader(f)

        for param in params:
            for i in range(100):

                # # tabular
                # start_time = time.time()
                # pred = sean(t_xx, t_txx, no_submodels = int(param["no_submodels"]), prep=[], extract=param["extract"])
                # end_time = time.time()
                # runtime = end_time - start_time
                # auc = roc_auc_score(t_tyy, pred)
                # logger.info(f'Tabular {param["no_submodels"]} \t {param["submodel"]} \t {param["prep"]} \t {param["extract"]} \t {runtime} \t {auc}')

                # image 
                start_time = time.time()
                pred = sean(i_xx, i_txx, no_submodels = int(param["no_submodels"]), prep=[], extract=param["extract"])
                end_time = time.time()
                runtime = end_time - start_time
                auc = roc_auc_score(i_tyy, pred)
                logger.info(f'Image {param["no_submodels"]} \t {param["submodel"]} \t {param["prep"]} \t {param["extract"]} \t {runtime} \t {auc}')

except Exception:
    logger.exception("message")



# # Replace dataset_id with the   ID of the dataset you want to load
# dataset = openml.datasets.get_dataset(
#     dataset_id= 42175,  # CreditCardFraudDetection
#     # dataset_id = 40900  # Satellite soil category
#     download_data=True,
#     download_qualities=True,
#     download_features_meta_data=True,
#     )

# X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# # Split the data into training and testing sets
# # test_size = 0.2  # Adjust the test size as needed (e.g., 0.2 for an 80-20 split)
# # random_state = 42  # You can set a random seed for reproducibility
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from sklearn.model_selection import train_test_split

# # Load CIFAR-10 data
# # CIFAR-10 contains 6000 images per class.
# # The original train-test split randomly divided these into 5000 train and 1000 test images per class.
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # normal class = 0
# # 10% anomaly
# index = np.where(y_train == 0)
# xx = x_train[index]
# index = np.where(y_train != 0)
# temp = x_train[index]
# sample_size = int(0.1*xx.shape[0])
# xx=np.append(xx, temp[np.random.choice(len(temp), size = sample_size, replace=False)], axis=0)

# index = np.where(y_test == 0)
# txx = x_test[index]
# tyy = np.zeros(len(txx)) # not anomalies
# index = np.where(y_test != 0)
# temp1 = x_test[index]
# temp2 = np.ones(len(temp1)) # anomalies

# sample_size = int(0.1*txx.shape[0])
# txx = np.append(txx, temp1[np.random.choice(len(temp1), size = sample_size, replace=False)], axis=0)
# tyy = np.append(tyy, temp2[np.random.choice(len(temp2), size = sample_size, replace=False)], axis=0)

# # runtime, extract, auc, no_submodels, submodel, altnorm = [], [], [], [], [], []
# for config in configs:
#     logger.info(f'Calling sean(altnorm={config["altnorm"]}, no_submodels={config["no_submodels"]}, submodel={config["submodel"]}, prep=[], extract={config["extract"]})')
#     start_time = time.time()
#     # pred = sean(X_train, X_test, altnorm=config["altnorm"], no_submodels = config["no_submodels"], prep=[], extract=config["extract"])
#     pred = sean(xx, txx, altnorm=config["altnorm"], no_submodels = config["no_submodels"], prep=[], extract=config["extract"])
#     end_time = time.time()
#     runtime = end_time - start_time
#     # auc = roc_auc_score(y_test, pred)
#     auc = roc_auc_score(tyy, pred)
#     logger.info(f'Result:: AUC: {auc} Runtime: {time.strftime("%H:%M:%S", time.gmtime(runtime))} \n')
# # print('\nROC_AUC: {} '.format(roc_auc_score(y_test, pred)))

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
