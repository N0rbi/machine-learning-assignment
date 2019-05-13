import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

class Encode:

    def __init__(self, dim_red_health, use_onehots):
        self.dim_red_health = dim_red_health
        self.use_onehots = use_onehots
        self.__create_transformer()

    def __create_transformer(self):
        self.__transformers = [
                ("type_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [0]),
                ("age_min_max", pp.MinMaxScaler(), [1]),
                ("breed_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [2,3]),
                ("gender_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [4]),
                ("color_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [5,6,7]),
                ("maturity_min_max", pp.MinMaxScaler(), [8]),
                ("fur_min_max", pp.MinMaxScaler(), [9]),
                ("health_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [10]),
                ("quantity_std", pp.StandardScaler(), [11]),
                ("fee_min_max", pp.MinMaxScaler(), [12]),
                ("state_onehot", pp.OneHotEncoder(handle_unknown="ignore"),[13]),
                ("photo_std", pp.StandardScaler(), [14]),
                ("vaccinated_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [15]),
                ("dewormed_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [16]),
                ("sterilized_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [17])
            ] if not self.dim_red_health else \
            [
                ("type_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [0]),
                ("age_min_max", pp.MinMaxScaler(), [1]),
                ("breed_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [2,3]),
                ("gender_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [4]),
                ("color_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [5,6,7]),
                ("maturity_min_max", pp.MinMaxScaler(), [8]),
                ("fur_min_max", pp.MinMaxScaler(), [9]),
            #     ("health_bulk_onehot", pp.StandardScaler(), [10]), PCA takes care of it
                ("health_onehot", pp.OneHotEncoder(handle_unknown="ignore"), [11]),
                ("quantity_std", pp.StandardScaler(), [12]),
                ("fee_min_max", pp.MinMaxScaler(), [13]),
                ("state_onehot", pp.OneHotEncoder(handle_unknown="ignore"),[14]),
                ("photo_std", pp.StandardScaler(), [15])
            ]
        if not self.use_onehots:
            self.__transformers = list(filter(lambda transform: "onehot" not in transform[0], self.__transformers))

    def get_column_transformer(self):
        return ColumnTransformer(self.__transformers)

def ann(train_X, train_y, test_X, test_y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 3), random_state=1)
    clf.fit(train_X, train_y)
    print("randomforest train acc:",clf.score(train_X,train_y))
    print("randomforest test acc:",clf.score(test_X,test_y))
    test_y_pred = clf.predict(test_X)
    accuracy = np.mean(test_y_pred.ravel() == test_y.ravel())
    print("Accuracy: " + str(accuracy))

    params = {
        'activation' : ['relu'],
        'solver': ['sgd', 'lbfgs', 'adam'],
        'batch_size': [200, 300, 400],
        'random_state': [1],
        'hidden_layer_sizes': [(16,3), (32, 3), (64, 3)]
    }

    clf_grid = GridSearchCV(MLPClassifier(), param_grid=params)
    clf_grid.fit(train_X, train_y)

    print("randomforest with gridsearchCV train acc:", clf_grid.score(train_X, train_y))
    print("randomforest with gridsearchCV test acc:", clf_grid.score(test_X, test_y))

    test_y_pred_grid = clf_grid.predict(test_X)
    grid_accuracy = np.mean(test_y_pred_grid.ravel() == test_y.ravel())
    print("Accuracy: " + str(grid_accuracy))
    print(clf_grid.best_estimator_)

    return (accuracy, grid_accuracy)

def get_data(dim_red_health=True):

    df = pd.read_csv("data/train/train.csv")
    reduced_df = df[[
    "Type", "Age", "Breed1", "Breed2",
    "Gender", "Color1", "Color2", "Color3",
    "MaturitySize", "FurLength", "Vaccinated",
    "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State",
    "PhotoAmt", "AdoptionSpeed"]]


    if dim_red_health:
        from sklearn.decomposition import PCA

        reduced_df_no_dim_red = reduced_df.copy()

        high_correlation_df = reduced_df[["Vaccinated", "Dewormed", "Sterilized"]]
        pca = PCA(n_components=1)
        pca.fit(high_correlation_df)

        # Seeing the high correlation between the 3 variables, we combine them
        del reduced_df["Vaccinated"]
        del reduced_df["Dewormed"]
        del reduced_df["Sterilized"]

        reduced_df["Health Stats Bulk"] = pd.Series(pca.transform(high_correlation_df).reshape(1,-1)[0])
        train_X = reduced_df[["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Health Stats Bulk", "Health", "Quantity", "Fee", "State", "PhotoAmt"]].values
    else:
        train_X = reduced_df[["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State", "PhotoAmt"]].values


    ct = Encode(dim_red_health, True).get_column_transformer()

    train_X = ct.fit_transform(train_X)

    train_y = reduced_df[["AdoptionSpeed"]].values

    return train_X, train_y


if __name__ == "__main__":
    dim_red_health = True

    df = pd.read_csv("data/train/train.csv")
    reduced_df = df[[
    "Type", "Age", "Breed1", "Breed2",
    "Gender", "Color1", "Color2", "Color3",
    "MaturitySize", "FurLength", "Vaccinated",
    "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State",
    "PhotoAmt", "AdoptionSpeed"]]

    reduced_df[["Vaccinated", "Dewormed", "Sterilized"]].corr()


    if dim_red_health:
        from sklearn.decomposition import PCA

        reduced_df_no_dim_red = reduced_df.copy()

        high_correlation_df = reduced_df[["Vaccinated", "Dewormed", "Sterilized"]]
        pca = PCA(n_components=1)
        pca.fit(high_correlation_df)

        # Seeing the high correlation between the 3 variables, we combine them
        del reduced_df["Vaccinated"]
        del reduced_df["Dewormed"]
        del reduced_df["Sterilized"]

        reduced_df["Health Stats Bulk"] = pd.Series(pca.transform(high_correlation_df).reshape(1,-1)[0])



    train_X = reduced_df[["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Health Stats Bulk", "Health", "Quantity", "Fee", "State", "PhotoAmt"]].values

    ct = Encode(dim_red_health, True).get_column_transformer()

    train_X = ct.fit_transform(train_X)

    train_y = reduced_df[["AdoptionSpeed"]].values
    #y_encoder = pp.OneHotEncoder(handle_unknown="ignore")
    #train_y = y_encoder.fit_transform(train_y).toarray()


    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)

    feature_num = train_X.shape[-1]
    record_num = train_X.shape[0]

    n_classes = np.unique(train_y).shape[0]

    ann(train_X, train_y, test_X, test_y)
