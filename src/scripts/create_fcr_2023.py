#%%
import pandas as pd

files = [
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-01-01_2023-01-31.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-02-01_2023-02-28.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-03-01_2023-03-31.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-04-01_2023-04-30.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-05-01_2023-05-31.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-06-01_2023-06-30.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-07-01_2023-07-31.xlsx",
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-08-01_2023-08-31.xlsx",
]

# import all files in pandas and expoert to csv

df = pd.DataFrame()
for file in files:
    df = df.append(pd.read_excel(f"data/{file}", engine="openpyxl"))

print(df.shape)

df.to_csv("data/fcr_2023.csv", index=False)
