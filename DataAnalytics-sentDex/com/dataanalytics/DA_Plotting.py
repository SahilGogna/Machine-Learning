# 2 - Using data plotting libraries to process the data

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/sahilgogna/Documents/ML/DataAnalytics-sentDex/data/avocado.csv")
df['Date'] = pd.to_datetime(df["Date"])
albany_df = df[df['region'] == "Albany"]
albany_df = albany_df.set_index("Date")
# albany_df["AveragePrice"].plot()
# plt.show()
albany_df = albany_df.sort_index()

# albany_df["AveragePrice"].rolling(25).mean().plot()
# plt.show()

albany_df['AveragePrice25'] = albany_df["AveragePrice"].rolling(25).mean()
# print(albany_df.head())
# initially we see Nans, because initially we need 25 values to calculate mean

# remove any value that has NaN
# albany_df.dropna().head(3)

# we can make a copy and save for future use
# albany_df = df.copy(df["region"] == "Albany")
# albany_df = albany_df.set_index("Date")
# albany_df = albany_df.sort_index()
# albany_df['AveragePrice25'] = albany_df["AveragePrice"].rolling(25).mean()

# if we want to convert into array of values
# this gives all the values including duplicates
df["region"].values.tolist()

# to get unique convert into set, then to iterate again convert it into the list
df["region"].unique()

graph_df = pd.DataFrame()

# for region in df['region'].unique()[:16]:
#     print(region)
#     region_df = df.copy()[df['region']==region]
#     region_df.set_index('Date', inplace=True)
#     region_df.sort_index(inplace=True)
#     region_df[f"{region}_price25ma"] = region_df["AveragePrice"].rolling(25).mean()
#
#     if graph_df.empty:
#         graph_df = region_df[[f"{region}_price25ma"]]  # note the double square brackets!
#     else:
#         graph_df = graph_df.join(region_df[f"{region}_price25ma"])

# https://pythonprogramming.net/graph-visualization-python3-pandas-data-analysis/

