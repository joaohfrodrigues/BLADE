# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:16:22 2019

@author: joaor
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "categorical_2"
sep_symbol = ";"
file_ext = "pdf"

sns.set(style="whitegrid")
df = pd.read_csv("accuracy/"+filename+".csv", index_col=False, sep=sep_symbol, header=0)

#exclude tDBN results in an auxiliar data frame
df_blade= df[df.algorithm == "BLADE"]

#assess hidden units impact
#df_hidden = df_blade[df_blade.time_steps == 12]
ax_hidden = sns.barplot(x="hidden_units", y="accuracy", data=df_blade, capsize=.2, palette=sns.color_palette("Oranges_r", 3))
ax_hidden.set(ylabel="Accuracy",xlabel='Hidden units')
plt.show(ax_hidden)


ax_time = sns.barplot(x="time_steps", y="accuracy", data=df_blade, capsize=.2, palette=sns.color_palette("Blues", 3))
ax_time.set(ylabel="Accuracy",xlabel='Time steps')
plt.show(ax_time)

ax_hidden.figure.savefig('accuracy_comparison/acc_'+filename+'_hidden.'+file_ext)
ax_time.figure.savefig('accuracy_comparison/acc_'+filename+'_time.'+file_ext)