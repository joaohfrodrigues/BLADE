# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:16:22 2019

@author: joaor
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = {'ecg': 'cornflowerblue', 'arabic':'forestgreen', 'jap':'sandybrown'}
file_ext = "pdf"
filename = "real"
sep_symbol = ";"

sns.set(style="whitegrid")
df = pd.read_csv("accuracy/"+filename+".csv", index_col=False, sep=sep_symbol, header=0)


df_blade = df[df.algorithm == 'BLADE']
for dataset in np.unique(df.dataset.values):
    df_ind = df_blade[df_blade.dataset == dataset]
    ax_individual = sns.barplot(x="hidden_units", y="accuracy", data=df_ind, capsize=.2, color=colors[dataset])
    ax_individual.set(xlabel='Hidden units', ylabel='Accuracy', title = str(dataset))
    plt.show(ax_individual)
    ax_individual.figure.savefig('accuracy_comparison/acc_'+filename+'_'+dataset+'.'+file_ext)


hidden_units = ['0', '24']
df_global = df[df.hidden_units.isin(hidden_units)]
ax_global = sns.catplot(x="dataset", y="accuracy", data=df_global,kind="bar", hue="algorithm", height=4, aspect=1.5, capsize=.2, legend=False)
ax_global.set(xlabel='Dataset')
plt.legend(title='Algorithm')
plt.show(ax_global)


ax_global.savefig('accuracy_comparison/acc_'+filename+'_global.'+file_ext)