from sean import *

import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import csv 
import time 
import json
import os

import sys

def read_config(task_id):
    # Read parameters from CSV file
    with open('params/p_cifar_slurm.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        # reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == task_id:
                return row

def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Get task ID from SLURM
    task_id = int(sys.argv[1])

    # Read configuration from CSV file
    param = read_config(task_id)

    for c in range(10):
        for i in range(10):
            print(f'CIFAR-10 rbm-lin Class-{c}: {i} out of 10')
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
            print(f'{datetime.now()} CIFAR10 Class-{normal_class} \t {param["prep"]} \t {param["extract"]} \t {param["submodel"]} \t {ensembles_executed} \t {runtime} \t {auc} \t {param["interaction_terms_then_randomize"]}')

if __name__ == "__main__":
    main()
