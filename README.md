## Introduction
This is the code for my master's thesis: <a href="Master Thesis with affidavit.pdf">Anomaly Detection Using an Ensemble with Simple Sub-models, 2024</a>.
The algorithm explores the effectiveness of an ensemble of simple sub-models like linear regression in detecting anomalies.

## Results
The AUROCs of the runs are stored in this <a href="https://docs.google.com/spreadsheets/d/1lLax3dy0JjQOxW_wwGM35UwRHdO8CJlR9QSlvxNbVNc/edit?usp=sharing">Google Sheet</a>

## How to pull from the GitHub Container Registry:
```
docker pull ghcr.io/vanlalpeka/msc_thesis:latest
```

<!-- ## Parameters
- `submodels_no` (int): The number of submodels to create.
- `features_no` (int): The number of features to use in each submodel.
- `computation_budget` (int): The maximum time to run the algorithm in seconds.
- `features_bagging_ratio` (float): The ratio of features to use in feature bagging.
- `train_test_split_ratio` (float): The ratio of the dataset to use for training.
- `data_path` (str): The path to the dataset. -->

## Example
```
python main.py <dataset>
```
Four dataset options: mnist, cifar10, ccfraud, sattelite

The following command will run the algorithm for $mnist$ dataset referring the run parameters from /src/config/main.csv
```
python main.py mnist
```


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