import sys
import argparse
import numpy as np

# i.e. from adess.py file, which is in the same folder as this file, import the functions
from .adess import adess, list_of_strings, print_arguments, print_result

def main():

    parser = argparse.ArgumentParser(description="An ensemble of simple sub=models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--X_train", help="Training data", type=np.ndarray, required=True)
    parser.add_argument("--X_test", help="Testing data", type=np.ndarray, required=True)
    parser.add_argument("--feat_sel_percent", help="Feature selection percentage", type=float, default=0.2)
    parser.add_argument("--max_feats", help="Maximum number of features", type=int, default=50)
    parser.add_argument("--order", help="Degree of polynomials for feature bagging", type=int, default=2)
    parser.add_argument("--computation_budget", help="Computation budget in seconds", type=int, default=600)
    parser.add_argument("--no_submodels", help="Count of submodels in the ensemble", type=int, default=500)
    parser.add_argument("--prep", help="List of preprocessing options (choose one or many): [skel,canny,clahe,blur,augment,gray,norm,std,none]", type=list_of_strings, default=['norm'])
    parser.add_argument("--extract", help="Feature selection option (choose one): [rbm,tsne,pca,ica,nmf,ae,none]", type=str, default='ae')
    parser.add_argument("--submodel", help="Submodel type option (choose one): [lin,lasso,elastic,svm]", type=str, default='lin')
    args = parser.parse_args()
    print_arguments(args)

    pred, ensembles_executed = adess(args.X_train, args.X_test, args.feat_sel_percent, args.max_feats, args.order, args.computation_budget, args.no_submodels, args.prep, args.extract, args.submodel)
    print_result(pred, ensembles_executed)

if __name__ == '__main__':
    main()