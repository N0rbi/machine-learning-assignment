from decisiontree import decisiontree, get_data as decision_data
from gmm import gmm, get_data as gmm_data
from randomforest import randomforest, get_data as randomforest_data
from ann2 import ann, get_data as ann_data
from knn import knn, get_data as knn_data
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

methods = [
    {"model": gmm, "data": gmm_data},
    {"model": decisiontree, "data": decision_data},
    {"model": randomforest, "data": randomforest_data},
    {"model": ann, "data": ann_data},
    {"model": knn, "data": knn_data}
]

seeds = [1]

outfile = open("out_dim_red.csv", "w")

print(";".join(["model_name", "seed", "PCA", "value"]), file=outfile)

for seed in seeds:
    for method in methods:
        train_X_orig, train_y_orig = method["data"](False)
        train_X_red, train_y_red = method["data"](True)
        for X, y, dimred in zip([train_X_orig, train_X_red], [train_y_orig, train_y_red], [False, True]):
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
            _, gs_accuracy = method["model"](train_X, train_y, test_X, test_y)

            print(";".join([str(method["model"].__name__), str(seed), str(dimred), str(gs_accuracy)]), file=outfile)
outfile.close()
