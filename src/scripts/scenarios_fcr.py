#%%
from datetime import timedelta

import numpy as np
import pandas as pd

# prices
date_cols = ["DATE_FROM", "DATE_TO"]
df1 = pd.read_csv("data/fcr_2021.csv", parse_dates=date_cols)
df2 = pd.read_csv("data/fcr_2022.csv", parse_dates=date_cols)

dk1 = ["DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
dk2 = ["DENMARK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]

df1 = df1[date_cols + dk1]
df2 = df2[date_cols + dk2]

df1.rename(columns={dk1[0]: "DK_PRICE"}, inplace=True)
df2.rename(columns={dk2[0]: "DK_PRICE"}, inplace=True)

df = pd.concat([df1, df2]).sort_values(by=date_cols).reset_index(drop=True)

df.DK_PRICE = df.DK_PRICE.apply(
    lambda x: x.replace(",", ".").replace("-", "nan")
).astype(float)

df = df.dropna().reset_index(drop=True)

grouper = pd.Grouper(key="DATE_FROM")

dates = []
vals = []

for ts, g in df.groupby(grouper):
    length = g.shape[0]
    if length == 10:
        g = g[g.DK_PRICE > 0]
    length = g.shape[0]
    ar = np.empty(24)
    assert 24 % length == 0
    incr = 24 // length
    for i, v in enumerate(g.DK_PRICE.values):
        assert incr * (i + 1) <= 24
        ar[incr * i : incr * (i + 1)] = v
        for j in range(incr * i, incr * (i + 1)):
            dates.append(ts + timedelta(hours=j))
    vals.append(ar)

assert len(dates) == len(np.concatenate(vals))
assert len(np.unique(dates)) == len(dates)

df2 = pd.DataFrame({"Date": dates, "DK_PRICE": np.concatenate(vals)})

# assert all differences are 1 hour for columns "Date" in df2
assert (df2.Date.diff().dt.seconds.dropna() == 3600).all()

# grid frequency
col = "Unnamed: 0"
fr = pd.read_csv("data/germany_2020_07.csv", parse_dates=[col])
fr.rename(columns={col: "Date"}, inplace=True)

fr.head()
fr.Frequency.describe()

print(fr.shape)

# group frequency per minute
min1 = fr.groupby(pd.Grouper(key="Date", freq="1min")).mean().reset_index()
min1.Frequency.describe()  # mHz originally
