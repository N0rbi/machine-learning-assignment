import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer
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

dim_red_health = False

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

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

X = ct.fit_transform(train_X)

y = reduced_df[["AdoptionSpeed"]].values
y_encoder = pp.OneHotEncoder(handle_unknown="ignore")

print(train_test_split(X, y, test_size=0.2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(X_train, y_train)

tpot.score(X_test, y_test)

tpot.export('pipeline.py')
