import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

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

def gmm(train_X, train_y, test_X, test_y):

    n_classes = np.unique(train_y).shape[0]
    tuned_parameters = [{'covariance_type': ['spherical','diag','tied','full']}]
    scoring='accuracy'

    train_y = train_y.reshape(train_y.shape[0])

    means_init = np.array([train_X[train_y == i].mean(axis=0) for i in range(n_classes)])

    gmm = GaussianMixture(n_components=n_classes, means_init=means_init, max_iter=100)

    gmm.fit(train_X, train_y)

    clf = GridSearchCV(gmm, tuned_parameters, cv=5, scoring=scoring)

    clf.fit(train_X, train_y)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    test_y_pred = gmm.predict(test_X)
    accuracy = np.mean(test_y_pred.ravel() == test_y.ravel())
    grid_test_y_pred = clf.predict(test_X)
    grid_accuracy = np.mean(grid_test_y_pred.ravel() == test_y.ravel())
    print("Accuracy: " + str(accuracy))

    return (accuracy, grid_accuracy,)

def get_data():
    dim_red_health = False

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
        train_X = reduced_df[["Type", "Age", "Breed1", "Breed2",
        "Gender", "Color1", "Color2", "Color3",
        "MaturitySize", "FurLength", "Vaccinated",
        "Dewormed", "Sterilized", "Health",
        "Quantity", "Fee", "State",
        "PhotoAmt"]]


    ct = Encode(dim_red_health, False).get_column_transformer()

    train_X = ct.fit_transform(train_X)

    train_y = reduced_df[["AdoptionSpeed"]].values
    return train_X, train_y


if __name__ == "__main__":
    dim_red_health = False

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
        train_X = reduced_df[["Type", "Age", "Breed1", "Breed2",
        "Gender", "Color1", "Color2", "Color3",
        "MaturitySize", "FurLength", "Vaccinated",
        "Dewormed", "Sterilized", "Health",
        "Quantity", "Fee", "State",
        "PhotoAmt"]]


    ct = Encode(dim_red_health, False).get_column_transformer()

    train_X = ct.fit_transform(train_X)

    train_y = reduced_df[["AdoptionSpeed"]].values
    feature_num = train_X.shape[-1]
    record_num = train_X.shape[0]

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)

    print(gmm(train_X, train_y, test_X, test_y))
