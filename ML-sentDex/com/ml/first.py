import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL', authtoken='QxzeUxuFcYhsyJ5Dk5Z_')

# selecting the columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Now for making Regression algorithm work correctly we need to convert the
# redundant features to meaningful features

df['Hl_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'Hl_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# it might happen we have a missing field
df.fillna(-99999, inplace=True)

# math.ceil - Return the ceiling of x, the smallest integer greater than or equal to x
forecast_out = math.ceil(0.01*len(df))

print(df.head())
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

