#%%
# %reload_ext autoreload
# %autoreload 2
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.utils import _set_font_size

# matplotlib.pyplot.ion()
sns.set_theme()
sns.set(font_scale=1.5)

BASE_FOLDER = "tex/figures/"


df = pd.read_excel("data/Log 26.09-10.10.xlsx")
print(df.columns)
print(df.KW.describe())

x = df.TimeStr.tolist()

# Consumption for a zone is ~110 kW when only one contactor is on, and ~442.5 kW when both are on
contactors = [
    "Q_LowerZone_Group_3_Contactor_2",
    "Q_LowerZone_Group_4_Contactor_2",
    "Q_UpperZone_Group_1_Contactor_2",
    "Q_UpperZone_Group_2_Contactor_2",
]

no_lz = df[contactors[0]] + df[contactors[1]] == 0
no_uz = df[contactors[2]] + df[contactors[3]] == 0
one_uz = df[contactors[2]] + df[contactors[3]] == 1

uz_low_consumption = df[
    (df[contactors[2]] + df[contactors[3]] == 1) & (no_lz)
].KW.mean()
uz_high_consumption = df[
    (df[contactors[2]] + df[contactors[3]] == 2) & (no_lz)
].KW.mean()

lz_low_consumption = df[
    (df[contactors[0]] + df[contactors[1]] == 1) & (no_uz)
].KW.mean()
lz_high_consumption = (
    df[(df[contactors[0]] + df[contactors[1]] == 2) & (one_uz)].KW.mean()
    - uz_low_consumption
)


def lower_zone_kw(x: Tuple[int, int, int, int]) -> float:
    if sum(x[0:2]) == 1:
        return lz_low_consumption
    elif sum(x[0:2]) == 2:
        return lz_high_consumption
    else:
        return 0


def upper_zone_kw(x: Tuple[int, int, int, int]) -> float:
    if sum(x[2:4]) == 1:
        return uz_low_consumption
    elif sum(x[2:4]) == 2:
        return uz_high_consumption
    else:
        return 0


df["KW_LZ"] = df.apply(lambda x: lower_zone_kw(tuple(x[contactors])), axis=1)
df["KW_UZ"] = df.apply(lambda x: upper_zone_kw(tuple(x[contactors])), axis=1)

# assert that the rule has been applied correctly
assert all(
    df[(df[contactors[2]] + df[contactors[3]] == 1) & (no_lz)].KW_UZ
    == uz_low_consumption
)
assert all(
    df[(df[contactors[2]] + df[contactors[3]] == 2) & (no_lz)].KW_UZ
    == uz_high_consumption
)
assert all(
    df[(df[contactors[0]] + df[contactors[1]] == 1) & (no_uz)].KW_LZ
    == lz_low_consumption
)
assert all(
    df[(df[contactors[0]] + df[contactors[1]] == 2) & (one_uz)].KW_LZ
    == lz_high_consumption
)

# error analysis
error = df.KW_LZ + df.KW_UZ - df.KW
error.describe()
df[abs(error) > 50]
# error.plot()

#%%
# %matplotlib widget
# %matplotlib notebook
f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 18))
ax = ax.ravel()
ax[0].step(
    x,
    df.KW.tolist(),
    label="kW-data",
    linestyle="--",
    color="black",
    linewidth=6,
    alpha=0.7,
)
# ax[0].step(x, df["KW-reconstructed"], label="kW-reconstruct1")
ax[0].step(x, df.KW_LZ + df.KW_UZ, label="kW-LZ+kW-UZ")
ax[0].step(x, df.KW_LZ, label="kW-LZ", linewidth=2, color="red")
ax[0].step(x, df.KW_UZ, label="kW-UZ", linewidth=2, color="green")
ax[0].set_title("Power consumption")
# ax[0].set_title(
#     "KW lower and upper zone | RMSE: {:.2f} | MAE: {:.2f}".format(rmse, mae)
# )
ax[0].set_ylabel("kW")
ax[1].plot(x, df.Temp_Upp_Dec, label=r"[$^\circ$C]")
ax[1].plot(
    x, df.SetpunktOppe, label="Setpoint", linestyle="--", alpha=0.5, color="black"
)
# ax[1].plot(x, np.concatenate(tz), label="TZ-hat", color="purple", linestyle="--")
ax2 = ax[1].twinx()
ax2.step(
    x,
    df.Q_UpperZone_Group_1_Contactor_2.tolist(),
    label="QU1",
    color="red",
    alpha=0.5,
    linestyle="--",
)
ax2.step(
    x,
    df.Q_UpperZone_Group_2_Contactor_2.tolist(),
    label="QU2",
    color="green",
    alpha=0.5,
    linestyle="--",
)
ax2.set_ylim([0, 5])
ax2.legend(loc="lower left")
ax2.tick_params(length=2)
ax2.set_yticklabels(["OFF", "ON"])

ax[1].set_title(r"Upper zone")
ax[1].set_ylabel(r"[$^\circ$C]")
ax[2].plot(x, df.Temp_Down_Dec, label=r"[$^\circ$C]")
ax[2].plot(
    x, df.SetpunktNede, label="Setpoint", linestyle="--", alpha=0.5, color="black"
)
ax2 = ax[2].twinx()
ax2.step(
    x,
    df.Q_LowerZone_Group_3_Contactor_2.tolist(),
    label="QL3",
    color="red",
    alpha=0.5,
    linestyle="--",
)
ax2.step(
    x,
    df.Q_LowerZone_Group_4_Contactor_2.tolist(),
    label="QL4",
    color="green",
    alpha=0.5,
    linestyle="--",
)
ax2.set_ylim([0, 5])
ax2.legend(loc="lower left")
ax2.tick_params(length=2)
ax2.set_yticklabels(["OFF", "ON"])

ax[2].set_title(r"Lower zone")
ax[2].set_ylabel(r"[$^\circ$C]")
ax[0].legend(loc="upper left")
ax[1].legend(loc="upper right")
ax[2].legend(loc="upper right")
start = 6400
end = 7800
ax[0].set_xlim([x[start], x[end]])
ax[2].xaxis.set_tick_params(rotation=45)

_set_font_size(ax)

plt.savefig(BASE_FOLDER + "data_visualization.png", dpi=300)

plt.show()

#%%
# Prepare data for parameter estimation in CTSM-R

##### Get regimes for upper zone #####
# regime: 1 if dipping as detected by period < 30 minuutes of temperature cycle using FFT
# regime: 2 if not dipping


def extract_period(data, sampling_rate, coeff: int = 5):
    assert coeff > 0
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(
        len(data), d=1
    )  # cycles per minut. d = spacing between samples in minutes

    if freqs[np.argmax(np.abs(fft_data))] == 0.0:
        # infinite period of signal. Happens when signal is constant (as for dot nordic temps.)
        peak_coefficient = np.argsort(np.abs(fft_data))[-coeff - 1 : -1]
    else:
        peak_coefficient = np.argsort(np.abs(fft_data))[-coeff:]

    peak_freq = (
        freqs[peak_coefficient]
        # * np.abs(fft_data)[peak_coefficient]
        # / sum(np.abs(fft_data)[peak_coefficient])
    )

    return 1 / abs(peak_freq / sampling_rate)


period_UZ = np.empty(df.shape[0])
period_UZ[0:200] = np.nan
period_LZ = np.empty(df.shape[0])
period_LZ[0:200] = np.nan

for i in range(200, df.shape[0]):
    p_u = extract_period(df.Temp_Upp_Dec.values[i - 120 : i], 1, 5)
    period_UZ[i] = p_u[-1]
    p_u = extract_period(df.Temp_Down_Dec.values[i - 120 : i], 1, 5)
    period_LZ[i] = p_u[-1]

df["period_UZ"] = period_UZ
df["period_LZ"] = period_LZ
cutoff_uz = 30  # minutes
cutoff_lz = 60  # minutes
df["regime_UZ"] = df.period_UZ.apply(lambda x: 0 if x > cutoff_uz else 1)
df["regime_LZ"] = df.period_LZ.apply(lambda x: 0 if x > cutoff_uz else 1)
df["regime_LZ"] = df["regime_UZ"].values  # we know the dipping from UZ...
df["t"] = np.arange(1 / 60, df.shape[0] / 60 + 1 / 60, 1 / 60)

# write to csv
cols = [
    "t",
    "TimeStr",
    "Temp_Upp_Dec",
    "Temp_Down_Dec",
    "regime_UZ",
    "regime_LZ",
    "KW_UZ",
    "KW_LZ",
]
df[cols].to_csv("data/dot_nordic_sde.csv", index=False)
