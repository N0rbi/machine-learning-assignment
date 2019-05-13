import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

r=pd.read_csv('out_dim_red.csv', sep=";")
sns.set(font_scale=2)
g=sns.catplot(data=r, col='model_name', col_wrap=3, y='value', sharey=False,
                  hue='PCA', kind='bar', x='seed', legend_out=False)
g.set_xlabels('').set_titles("{col_name}")
plt.tight_layout()
plt.savefig("models_dim.pdf")
plt.savefig("models_dim.png")
plt.close()
