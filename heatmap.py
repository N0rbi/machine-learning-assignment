import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("data/train/train.csv")
reduced_df = df[["Type", "Age", "Breed1", "Breed2",
"Gender", "Color1", "Color2", "Color3", "MaturitySize",
"FurLength", "Vaccinated", "Dewormed",
"Sterilized", "Health", "Quantity",
 "Fee", "State", "PhotoAmt", "AdoptionSpeed"]]

import seaborn as sns
import matplotlib.pyplot as plt

corr = reduced_df.corr()

# plot the heatmap
sns.heatmap(corr.abs(),
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.savefig("heatmap.png")
plt.savefig("heatmap.pdf")
