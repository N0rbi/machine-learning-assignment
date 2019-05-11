import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
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
# y_encoder = pp.OneHotEncoder(handle_unknown="ignore")
# train_y = y_encoder.fit_transform(train_y).toarray()


from keras.models import Sequential
from keras.layers import Dense, Dropout

feature_num = train_X.shape[-1]
record_num = train_X.shape[0]

label_feature_num = train_y.shape[-1]

dense_layers = [
    (500, 'relu'),
    (750, 'relu'),
    (300, 'relu'),
]

print(train_X.shape)

model = Sequential()

model.add(Dense(600, input_dim=feature_num, name="input_dense"))
for i, (layer_units, layer_actication_type) in enumerate(dense_layers):
    model.add(Dense(layer_units, activation=layer_actication_type, name="hidden_dense_%d"%i))
    model.add(Dropout(.2))

model.add(Dense(label_feature_num, activation="softmax", name="output_dense"))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)

model.fit(train_X, train_y, epochs=100, batch_size=record_num//10, validation_split = 0.2)

def get_build_model(feature_num, label_feature_num):
    def build_model(optimizer, dense_layers):
        model = Sequential()
        model.add(Dense(600, input_dim=feature_num, name="input_dense"))
        for i, (layer_units, layer_actication_type) in enumerate(dense_layers):
            model.add(Dense(layer_units, activation=layer_actication_type, name="hidden_dense_%d"%i))
            model.add(Dropout(.2))

        model.add(Dense(label_feature_num, activation="softmax", name="output_dense"))
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    return build_model

classifier = KerasClassifier(build_fn= get_build_model(feature_num, label_feature_num), epochs=100)

tuned_parameters = {"optimizer": ["adam", "SGD"], "dense_layers": [[(200, "relu"), (200, "relu")], [(400, "relu")]]}
clf_grid = GridSearchCV(classifier, tuned_parameters, cv=5, scoring="accuracy")

clf_grid.fit(train_X, train_y)

test_y_pred = model.predict(test_X)
accuracy = np.mean(test_y_pred.ravel() == test_y.ravel())
print("Accuracy: " + str(accuracy))

test_y_pred_grid = clf_grid.predict(test_X)
grid_accuracy = np.mean(test_y_pred_grid.ravel() == test_y.ravel())
print("Accuracy: " + str(grid_accuracy))
print(clf_grid.best_estimator_)

# return (accuracy, grid_accuracy)
