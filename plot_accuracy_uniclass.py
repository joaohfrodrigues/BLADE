# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:16:22 2019

@author: joaor
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "uniclass"
sep_symbol = ";"
file_ext = "pdf"

sns.set(style="whitegrid")
df = pd.read_csv("accuracy/"+filename+".csv", index_col=False, sep=sep_symbol, header=0)

#ax_compare = sns.catplot(x="dataset", y="accuracy", hue='algorithm', kind="bar", data=df, height=4, aspect=2, capsize=.1, legend_out = True)
ax_compare = sns.catplot(x="dataset_alt", y="accuracy", hue='algorithm', kind="bar", data=df, height=4, aspect=1.6, capsize=.1, legend = False)
ax_compare.set(ylabel="Accuracy",xlabel='Dataset')
#ax_compare._legend.set_title('Algorithm')
plt.legend(title='Algorithm')
plt.legend(loc='upper right')
plt.show(ax_compare)

ax_compare.savefig('accuracy_comparison/acc_'+filename+'_compare.'+file_ext)