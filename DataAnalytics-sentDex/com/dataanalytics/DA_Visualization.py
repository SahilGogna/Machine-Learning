# 4 - visualizing the correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

df = pd.read_csv("/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/minwage.csv")
act_min_wage = pd.DataFrame

for name, group in df.groupby("State"):
    if act_min_wage.empty:
        act_min_wage = group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018": name})
    else:
        act_min_wage = act_min_wage.join(group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018": name}))

min_wage_corr = act_min_wage.replace(0, np.nan).dropna(axis=1).corr()

# plt.matshow(min_wage_corr)
# plt.show()

labels = [c[:2] for c in min_wage_corr.columns]  # get abbv state names.

fig = plt.figure(figsize=(12,12))  # figure so we can add axis
ax = fig.add_subplot(111)  # define axis, so we can modify
ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)  # display the matrix
ax.set_xticks(np.arange(len(labels)))  # show them all!
ax.set_yticks(np.arange(len(labels)))  # show them all!
ax.set_xticklabels(labels)  # set to be the abbv (vs useless #)
ax.set_yticklabels(labels)  # set to be the abbv (vs useless #)

# plt.show()

web = requests.get("https://www.infoplease.com/state-abbreviations-and-state-postal-codes")
dfs = pd.read_html(web.text)

# for df in dfs:
#     print(df.head())

state_abbv = dfs[0]

# run once to save it to the file
# state_abbv.to_csv("/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/state_abbv.csv")

# state_abbv = pd.read_csv("/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/state_abbv.csv")
# we accidentally saved the index also which is useless
# print(state_abbv.head())

# read again without the index state_abbv[["State/District", "Postal Code"]].to_csv(
# "/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/state_abbv.csv", index=False)  # index in this case is
# worthless
state_abbv = pd.read_csv("/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/state_abbv.csv", index_col=0)
# print(state_abbv.head())

abbv_dict = state_abbv.to_dict()
abbv_dict = abbv_dict['Postal Code']

abbv_dict['Federal (FLSA)'] = "FLSA"
abbv_dict['Guam'] = "GU"
abbv_dict['Puerto Rico'] = "PR"
labels = [abbv_dict[c] for c in min_wage_corr.columns]  # get abbv state names.

fig = plt.figure(figsize=(12,12))  # figure so we can add axis
ax = fig.add_subplot(111)  # define axis, so we can modify
ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)  # display the matrix
ax.set_xticks(np.arange(len(labels)))  # show them all!
ax.set_yticks(np.arange(len(labels)))  # show them all!
ax.set_xticklabels(labels)  # set to be the abbv (vs useless #)
ax.set_yticklabels(labels)  # set to be the abbv (vs useless #)

plt.show()