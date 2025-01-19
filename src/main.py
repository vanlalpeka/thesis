#######################################################################################################################
# Say, the ensemble size is 50.
# This function will call pre_process() and feature_selection() once each, 
# and then call one_model() 50 times.
#
# one_model() handles both feature bagging and the sub-model.
#######################################################################################################################

import numpy as np
import pandas as pd
import csv 
import time 
import os
import sys
import logging 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import cifar10, mnist


from pre_process import *
from feature_selection import *
from feature_bagging import *
from one_model import *
from sean import *

def main(dataset_id='ccfraud'):
    if not dataset_id:
        raise ValueError("dataset_id is required")
    
    # create log directory if it does not exist
    os.makedirs("log/", exist_ok=True)

    logging.basicConfig(filename=f"log/AUROC_{dataset_id}.log", 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 

    #Let us Create an object 
    logger=logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 

    print("START")
    logger.info('START')

    try:
        with open("config/main.csv") as f:
            params = csv.DictReader(f, delimiter=';')

            # tabular datasets
            if dataset_id in ['ccfraud', 'sattelite']:
                X, y = fetch_openml('dataset_id', version=1, as_frame=True)

                # Identify indices of samples where y=1 (fraudulent transactions)
                fraud_indices = [i for i, label in enumerate(y) if label == 1]

                X_fraud = X.loc[fraud_indices]
                y_fraud = y.loc[fraud_indices]

                X_no_fraud = X.drop(fraud_indices)
                y_no_fraud = y.drop(fraud_indices)

                # Split the data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X_no_fraud, y_no_fraud, test_size=0.2)

                # Include all samples with y=1 in the test set
                # In other words, the training set has no anomalies
                X_test = pd.concat([X_test, X_fraud], axis=0)
                y_test = pd.concat([y_test, y_fraud], axis=0)

                for param in params:
                    for i in range(10):
                        start_time = time.time()
                        print(f'dataset_id: {dataset_id} | {i} out of 10 | param: {param}')

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
                        print(f'AUROC : {auc} ')
                        logger.info(f'{dataset_id} \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc}')

            # image datasets
            elif dataset_id in ['mnist', 'cifar10']:
                (x_train, y_train), (x_test, y_test) = mnist.load_data() if dataset_id == 'mnist' else cifar10.load_data()
                for param in params:
                    for c in range(10): # 10 classes
                        for i in range(10): # Run 10 times for each class
                            print(f'dataset_id: {dataset_id} | Class:{c} | {i} out of 10 | param: {param}')
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
                            logger.info(f'{dataset_id}  Class-{normal_class} \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc}')
            
            # invalid dataset_id
            else:
                raise ValueError("Invalid dataset_id")

    except Exception:
        logger.exception("message")

    logger.info('END')


# executes only when run directly, not when this file is imported into another python file
if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(dataset_id=sys.argv[1]) 
    else:
        main() 
