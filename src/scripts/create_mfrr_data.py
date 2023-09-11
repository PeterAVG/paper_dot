#%%
import numpy as np
import pandas as pd

DK = "DK1"

file = "mfrrreservesdk2.csv" if DK == "DK2" else "mfrrreservesdk1.csv"
file = "/Users/petergade/Box Sync/pega/Data/prices/" + file
mfrr = pd.read_csv(file, sep=";", decimal=",")
print(mfrr.shape)

file = "RegulatingBalancePowerdata.csv"
file = "/Users/petergade/Box Sync/pega/Data/prices/" + file
rt = pd.read_csv(file, sep=";", decimal=",")
rt = rt.query(f"PriceArea == '{DK}'")
rt.rename(
    columns={
        "mFRRUpActBal": "RegulatingPowerUp",
        "mFRRDownActBal": "RegulatingPowerDown",
    },
    inplace=True,
)
print(rt.shape)

file = "elspotprices.csv"
file = "/Users/petergade/Box Sync/pega/Data/prices/" + file
elspot = pd.read_csv(file, sep=";", decimal=",")
elspot = elspot.query(f"PriceArea == '{DK}'")
print(elspot.shape)

df = pd.merge(mfrr, rt, left_on="HourUTC", right_on="HourUTC", how="outer")
df = pd.merge(df, elspot, left_on="HourUTC", right_on="HourUTC", how="left")
df.sort_values(by="HourUTC", inplace=True, ascending=True)
df.reset_index(drop=True, inplace=True)
df.set_index("HourUTC", inplace=True, drop=False)
df = df.query(
    # "HourUTC >= '2020-08-03'"
    "HourUTC >= '2021-01-01'"
)  # '2020-08-03' is earliest date with spot prices

print(df.shape)

columns = [
    "HourUTC",
    # "mFRR_UpPurchased",
    "mFRR_UpPriceDKK",
    "RegulatingPowerUp",
    "RegulatingPowerDown",
    "BalancingPowerPriceUpDKK",
    "BalancingPowerPriceDownDKK",
    "SpotPriceDKK",
]


mf = df[columns].dropna()
mf["flag"] = (mf.RegulatingPowerUp > 0).astype(int)
mf["flag_down"] = (mf.RegulatingPowerDown < 0).astype(int)
mf["flag_sum"] = mf.flag.cumsum()
mf["flag_sum_diff"] = mf.flag_sum.diff()
mf["flag_sum_mwh"] = mf.RegulatingPowerUp.cumsum()
mf["flag_sum"]


mf[["flag", "flag_sum", "flag_sum_diff", "flag_sum_mwh"]].iloc[0:40]

# NOTE: remember this very handy line for getting consecutive groups!
mf["group"] = mf["flag"].ne(mf["flag"].shift()).cumsum()

mf[["flag", "group"]].iloc[0:40]


#### SCENARIOS VERSION 2 ####
tmp = mf[
    [
        "HourUTC",
        "flag",
        "flag_down",
        "SpotPriceDKK",
        "BalancingPowerPriceUpDKK",
        "BalancingPowerPriceDownDKK",
        "mFRR_UpPriceDKK",
    ]
]
tmp["Hour"] = pd.to_datetime(tmp.HourUTC).dt.hour
tmp["Date"] = pd.to_datetime(tmp.HourUTC).dt.date
print(tmp.shape)
tmp.dropna(axis=1, how="any", inplace=True)
print(tmp.shape)
tmp.isna().sum().sort_values(ascending=False)
tmp.to_csv(
    f"data/scenarios_v3_{DK}.csv",
    index=False,
)
