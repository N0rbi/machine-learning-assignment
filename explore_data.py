import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# breeds = pd.read_csv('../input/breed_labels.csv')
# colors = pd.read_csv('../input/color_labels.csv')
# states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('data/train/train.csv')
test = pd.read_csv('data/test/test.csv')
# sub = pd.read_csv('../input/test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])

columns = [
"Age", "Gender",
"MaturitySize", "FurLength", "Vaccinated",
"Dewormed", "Sterilized", "Health",
"Quantity", "Fee",
"PhotoAmt"]

target = "AdoptionSpeed"

fig = plt.figure()

train_folded_all = pd.DataFrame()

train_less_col = train[columns]

for column in columns:
    train_folded = train[[target]]
    train_folded["value"] = train[[column]]
    train_folded["column"] = column
    train_folded_all = pd.concat([train_folded_all, train_folded])

train_folded_all[target] =  train_folded_all[target].astype('category')

# sns.catplot(data=train_folded_all, col="column", y="value", sharey=False, x="AdoptionSpeed",  kind="count")

categorical = ["Gender", "Vaccinated",
"Dewormed", "Sterilized", "Health", "MaturitySize"]

non_categorical = ["Age", "FurLength", "Quantity", "Fee",
"PhotoAmt"]

for name, column, kind in (zip(["cat", "non_cat"], [categorical, non_categorical], ["count", "violin"])):
    sns.catplot(data=train_folded_all[train_folded_all["column"].isin(column)], col="column", y="value", sharey=False,
     x="AdoptionSpeed" if name == "non_cat" else None,  kind=kind)
    plt.tight_layout()
    plt.savefig("boxes_%s.png" % name)


# source: https://www.kaggle.com/artgor/exploration-of-data-step-by-step

train['AdoptionSpeed'].value_counts().sort_index().plot('barh', color='teal');
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'])
plt.title('Adoption speed classes rates');
ax=g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')
plt.savefig("train_classes_percent.png")


all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
plt.figure(figsize=(10, 6));
sns.countplot(x='dataset_type', data=all_data, hue='Type');
plt.title('Number of cats and dogs in train and test data');
plt.savefig("cat_dog_count.png")
