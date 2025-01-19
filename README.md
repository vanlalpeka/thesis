## Introduction
This is the code for my master's thesis: <a href="Master Thesis with affidavit.pdf">Anomaly Detection Using an Ensemble with Simple Sub-models, 2024</a>.
The algorithm explores the effectiveness of an ensemble of simple sub-models like linear regression in detecting anomalies.

## Usage
The main.py file contains the main function to run the algorithm. The algorithm is designed to run for four datasets: mnist, cifar10, ccfraud, and sattelite. The main function will run the algorithm for the dataset provided as a command-line argument. The run parameters for the algorithm are stored in /src/config/main.csv. The results of the run are stored in /src/log. The tabular datasets are pulled from OpenML at runtime, and the image datasets are tensorflow built-in datasets.

## Datasets
- MNIST:
- CIFAR10: 
- CCFraud:
- Satellite: 

## CLI Parameters
The parameters are as follows:
- dataset_id: the dataset to use (mnist, cifar10, ccfraud, sattelite)
- feat_sel_percent: the percentage of features to use for feature selection
- max_feats: the maximum number of features to use
- order: the order of the submodel (1, 2, 3)
- computation_budget: the computation budget in seconds

## Example
```
python main.py --dataset_id=mnist
```

## Results
The AUROCs of the runs reported in the thesis are stored in this <a href="https://docs.google.com/spreadsheets/d/1lLax3dy0JjQOxW_wwGM35UwRHdO8CJlR9QSlvxNbVNc/edit?usp=sharing">Google Sheet</a>


## Pseudo-code
```
train_test_split()
for _ in range(10):
    pre_process()
    feature_selection()
    check_computation_budget()  #  default: 600 seconds

    for _ in submodels_no:
        feature_bagging()  # feature bagging with no interaction terms
        one_submodel()  # the ensemble
        check_computation_budget()  #  default: 600 seconds
        roc_auc_score()

```