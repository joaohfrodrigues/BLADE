# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:16:22 2019

@author: joaor
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "multiclass"
sep_symbol = ";"
file_ext = "pdf"

sns.set(style="whitegrid")
df = pd.read_csv("accuracy/"+filename+".csv", index_col=False, sep=sep_symbol, header=0)

#exclude tDBN results in an auxiliar data frame
df_blade= df[df.algorithm == "BLADE"]
df_blade_3= df_blade[df_blade.classes == 3]
df_blade_10= df_blade[df_blade.classes == 10]

#assess hidden units impact on 3 class dataset
ax_hidden = sns.barplot(x="hidden_units", y="accuracy", data=df_blade_3, capsize=.2, palette=sns.color_palette("Oranges", 6))
ax_hidden.set(ylabel="Accuracy",xlabel='Hidden units')
plt.show(ax_hidden)
ax_hidden.figure.savefig('accuracy_comparison/acc_'+filename+'_hidden_3.'+file_ext)
#assess hidden units impact on 10 class dataset
ax_hidden = sns.barplot(x="hidden_units", y="accuracy", data=df_blade_10, capsize=.2, palette=sns.color_palette("Blues", 6))
ax_hidden.set(ylabel="Accuracy",xlabel='Hidden units')
plt.show(ax_hidden)
ax_hidden.figure.savefig('accuracy_comparison/acc_'+filename+'_hidden_10.'+file_ext)


#ax_compare = sns.catplot(x="dataset", y="accuracy", hue='algorithm', kind="bar", data=df, height=4, aspect=2, capsize=.1, legend_out = True)
ax_compare = sns.catplot(x="classes", y="accuracy", hue='algorithm', kind="bar", data=df, height=4, aspect=1.6, capsize=.1, legend = False)
ax_compare.set(ylabel="Accuracy",xlabel='Number of classes')
#ax_compare._legend.set_title('Algorithm')
plt.legend(title='Algorithm')
plt.legend(loc='upper left')
plt.show(ax_compare)

ax_compare.savefig('accuracy_comparison/acc_'+filename+'_compare.'+file_ext)