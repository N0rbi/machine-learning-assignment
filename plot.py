import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

r=pd.read_csv('out.csv', sep=";")

g=sns.catplot(data=r, col='model_name', col_wrap=3, y='value', sharey=False,
                  hue='gridSearch', kind='bar', x='seed', legend_out=False)
g.set_xlabels('').set_titles("{col_name}")
plt.tight_layout()
plt.savefig("models.pdf")
plt.close()
