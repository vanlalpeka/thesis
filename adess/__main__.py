import sys
import argparse
import numpy as np
import pathlib

# i.e. from adess.py file, which is in the same folder as this file, import the functions
from adess.adess import adess

def main():
    # print('__main__.py: main()')
    parser = argparse.ArgumentParser(description="ADESS: Anomaly Detection using Ensemble of simple sub=models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", help="Training data", type=pathlib.Path, required=True)
    parser.add_argument("--test", help="Testing data", type=pathlib.Path, required=True)
    parser.add_argument("--feat_sel_percent", help="Feature selection percentage", type=float, default=0.2)
    parser.add_argument("--max_feats", help="Maximum number of features", type=int, default=50)
    parser.add_argument("--order", help="Degree of polynomials for feature bagging", type=int, default=2)
    parser.add_argument("--computation_budget", help="Computation budget in seconds", type=int, default=600)
    parser.add_argument("--no_submodels", help="Count of submodels in the ensemble", type=int, default=500)
    parser.add_argument("--prep", help="List of preprocessing options (choose one or many): skel,canny,clahe,blur,augment,gray,norm,std,none", type=str, default="norm")
    parser.add_argument("--extract", help="Feature selection option (choose one): rbm,tsne,pca,ica,nmf,ae,none", type=str, default='pca')
    parser.add_argument("--submodel", help="Submodel type option (choose one): lin,lasso,elastic,svm", type=str, default='lin')
    args = parser.parse_args()

    X_train = np.load(args.train)
    X_test = np.load(args.test)

    f = [f'{k} = {v}' for k, v in vars(args).items()]
    print(f'X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}, {str(f[2:])[1:-1]}')

    pred, ensembles_executed = adess(X_train=X_train, X_test=X_test, 
                                     no_submodels=args.no_submodels, feat_sel_percent=args.feat_sel_percent, 
                                     max_feats = args.max_feats, order=args.order, prep=args.prep.split(','), 
                                     extract=args.extract, submodel=args.submodel, computation_budget=args.computation_budget)
    
    print(f'Mean of Predicted Y = {pred}, Count of submodel executed = {ensembles_executed}')


if __name__ == '__main__':
    main()