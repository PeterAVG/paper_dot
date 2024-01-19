#%%
from typing import Tuple

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.utils import _set_font_size

# import seaborn as sns


# matplotlib.pyplot.ion()
# sns.set_theme()
# sns.set(font_scale=1.5)

BASE_FOLDER = "tex/figures/"


###### SIMULATE 4th order model ######
###### SIMULATE NEW OPERATION WITH STEADY STATE POWER SETPOINTS ######

# parameters of physical model

# This model is relly good at transients around ix=[6400:7800]
ta = 20
Czu = 1.029
Czl = 0.502
Cwu = 6
Cwl = 6
Rww = 0.829
Rwz = 0.185
Rzuzl = 48
Rwua1 = 3.795
Rwua2 = 2.581
Rwla1 = 11.743
# Rwla2 = 4.646


df = pd.read_csv("data/dot_nordic_sde.csv")
regime = df.regime_UZ.values

dt_ = 1 / 60  # 1 minute

# low-power regime, r1
regime1 = f"regime_UZ == {0}"
# r1_kw_uz_ss = df.iloc[7500:8500].KW_UZ.mean()
# r1_kw_lz_ss = df.query(regime1).KW_LZ.mean()
low_power_setpoint = "KW_UZ < 150 & KW_UZ > 0"
r1_kw = df.query(low_power_setpoint).KW_UZ.mean()

# high-power regime, r2
regime2 = f"regime_UZ == {1}"
# r2_kw_uz_ss = df.iloc[1000:2000].KW_UZ.mean()
# r2_kw_lz_ss = df.query(regime2).KW_LZ.mean()
high_power_setpoint = "KW_UZ > 150"
r2_kw = df.query(high_power_setpoint).KW_UZ.mean()

twu_data = df.Temp_Upp_Dec.values
twl_data = df.Temp_Down_Dec.values

p_up_data = df.KW_UZ.values
p_low_data = df.KW_LZ.values

x = pd.to_datetime(df.TimeStr.tolist()).tolist()

setpoint_lz = 447 * np.ones(len(x))
setpoint_uz = 455 * np.ones(len(x))

# r1_kw_uz_ss = np.mean([p for i, p in enumerate(p_up_data) if regime[i] == 0])
# r2_kw_uz_ss = np.mean([p for i, p in enumerate(p_up_data) if regime[i] == 1])
# r1_kw_lz_ss = np.mean([p for i, p in enumerate(p_low_data) if regime[i] == 0])
# r2_kw_lz_ss = np.mean([p for i, p in enumerate(p_low_data) if regime[i] == 1])
r1_kw_uz_ss = (-Rww * 20 + 8 * Rwua1 + Rww * 455) / (Rwua1 * Rww)
r2_kw_uz_ss = (-Rww * 20 + 8 * Rwua2 + Rww * 455) / (Rwua2 * Rww)
r1_kw_lz_ss = -(Rww * 20 + 8 * Rwla1 - Rww * 447) / (Rwla1 * Rww)
r1_kw_uz_ss = (455 - 20) / Rwua1 + 8 / Rww
r2_kw_uz_ss = (455 - 20) / Rwua2 + 8 / Rww
r1_kw_lz_ss = (447 - 20) / Rwla1 - 8 / Rww

assert r1_kw_uz_ss is not None and r2_kw_uz_ss is not None and r1_kw_lz_ss is not None
print(r1_kw_uz_ss, r2_kw_uz_ss, r1_kw_lz_ss)


def simulate_4th_order_model(
    start: int,
    end: int,
    steady_state: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    Tzl_sim = np.empty(end - start)
    Tzu_sim = np.empty(end - start)
    Twl_sim = np.empty(end - start)
    Twu_sim = np.empty(end - start)
    Tzl_sim[0] = setpoint_lz[0]
    Tzu_sim[0] = setpoint_uz[0]
    Twl_sim[0] = twl_data[start:end][0]
    Twu_sim[0] = twu_data[start:end][0]
    p_up = np.empty(end - start)
    p_low = np.empty(end - start)
    p_up[0] = df.KW_UZ.values[start:end][0]
    p_low[0] = df.KW_LZ.values[start:end][0]

    if steady_state:
        Tzl_sim[0] = setpoint_lz[0]
        Tzu_sim[0] = setpoint_uz[0]
        Twl_sim[0] = setpoint_lz[0]
        Twu_sim[0] = setpoint_uz[0]
        p_up[0] = r1_kw_uz_ss if regime[start] == 0 else r2_kw_uz_ss
        p_low[0] = r1_kw_lz_ss

    if p_up[0] > 400:
        Q1_up, Q2_up = 1, 1
    elif p_up[0] <= 400 and p_up[0] > 90:
        Q1_up, Q2_up = 1, 0
    elif p_up[0] <= 90:
        Q1_up, Q2_up = 0, 0

    if p_low[1] > 400:
        Q1_low = 1
        Q2_low = 1
    elif p_low[1] <= 400 and p_low[1] > 90:
        Q1_low = 1
        Q2_low = 0
    elif p_low[1] <= 90:
        Q1_low = 0
        Q2_low = 0

    for i in range(1, end - start):
        j = start + i

        if Twu_sim[i - 1] < 454.5:
            Q1_up = 1
        elif Twu_sim[i - 1] > 456.5:
            Q1_up = 0
        if Twu_sim[i - 1] < 453.5:
            Q2_up = 1
        elif Twu_sim[i - 1] > 455.5:
            Q2_up = 0

        if Q1_up + Q2_up == 0:
            _p_up = 0.0
        elif Q1_up + Q2_up == 1:
            _p_up = r1_kw
        elif Q1_up + Q2_up == 2:
            _p_up = r2_kw
        else:
            raise ValueError("Something went wrong")

        if Twl_sim[i - 1] < 446.5:
            Q1_low = 1
        elif Twl_sim[i - 1] > 448.5:
            Q1_low = 0
        if Twl_sim[i - 1] < 445.5:
            Q2_low = 1
        elif Twl_sim[i - 1] > 447.5:
            Q2_low = 0

        if Q1_low + Q2_low == 0:
            _p_low = 0.0
        elif Q1_low + Q2_low == 1:
            _p_low = r1_kw
        elif Q1_low + Q2_low == 2:
            _p_low = r2_kw
        else:
            raise ValueError("Something went wrong")

        if steady_state:
            _p_up = r1_kw_uz_ss if regime[j] == 0 else r2_kw_uz_ss
            _p_low = r1_kw_lz_ss
            if i > 230 and i <= -850:
                # _p_up = 0
                _p_low = 0
            elif i > 360 and i <= 370:
                # _p_up = 440
                # _p_low = 440
                pass

        p_up[i] = _p_up
        p_low[i] = _p_low

        ####### 4th ORDER MODEL #######
        Tzu_sim[i] = Tzu_sim[i - 1] + 1 / Czu * dt_ * (
            1 / Rzuzl * (Tzl_sim[i - 1] - Tzu_sim[i - 1])
            + 1 / Rwz * (Twu_sim[i - 1] - Tzu_sim[i - 1])
        )
        Tzl_sim[i] = Tzl_sim[i - 1] + 1 / Czl * dt_ * (
            1 / Rzuzl * (Tzu_sim[i - 1] - Tzl_sim[i - 1])
            + 1 / Rwz * (Twl_sim[i - 1] - Tzl_sim[i - 1])
        )

        Twu_sim[i] = Twu_sim[i - 1] + 1 / Cwu * dt_ * (
            (1 - regime[j]) * 1 / Rwua1 * (ta - Twu_sim[i - 1])
            + regime[j] * 1 / Rwua2 * (ta - Twu_sim[i - 1])
            + 1 / Rww * (Twl_sim[i - 1] - Twu_sim[i - 1])
            + 1 / Rwz * (Tzu_sim[i - 1] - Twu_sim[i - 1])
            + _p_up
        )
        Twl_sim[i] = Twl_sim[i - 1] + 1 / Cwl * dt_ * (
            1 / Rwla1 * (ta - Twl_sim[i - 1])
            + 1 / Rww * (Twu_sim[i - 1] - Twl_sim[i - 1])
            + 1 / Rwz * (Tzl_sim[i - 1] - Twl_sim[i - 1])
            + _p_low
        )

    return Tzl_sim, Tzu_sim, Twl_sim, Twu_sim, p_up, p_low


start = 6400
end = 7840
# start = 1000
# end = 2000

Tzl_sim, Tzu_sim, Twl_sim, Twu_sim, p_up, p_low = simulate_4th_order_model(start, end)
(
    Tzl_sim_ss,
    Tzu_sim_ss,
    Twl_sim_ss,
    Twu_sim_ss,
    p_up_ss,
    p_low_ss,
) = simulate_4th_order_model(start, end, True)

# print sum of power vs p
print(f"UZ: Sum of power: {round(sum(p_up)/60,3)}")
print(f"UZ: Sum of pUpp data: {round(sum(p_up_data[start:end])/60,3)}")
print(
    f"UZ: Error: {round(abs((sum(p_up/60) - sum(p_up_data[start:end]/60)) / sum(p_up_data[start:end]/60))*100, 3)}%"
)

print(f"LZ: \nSum of power: {round(sum(p_low)/60,1)}")
print(f"LZ: Sum of pLow data: {round(sum(p_low_data[start:end])/60,3)}")
print(
    f"LZ: Error: {round(abs((sum(p_low)/60 - sum(p_low_data[start:end]/60)) / sum(p_low_data[start:end]/60))*100, 3)}%"
)

print("Saving simulation into file...")
tmp = df.iloc[start:end, :].copy()
tmp["KW_UZ_SS"] = p_up_ss
tmp["KW_LZ_SS"] = p_low_ss
tmp["Twu_sim_ss"] = Twu_sim_ss
tmp["Twl_sim_ss"] = Twl_sim_ss
# tmp[["regime_UZ", "KW_UZ_SS", "KW_LZ_SS", "Twu_sim_ss", "Twl_sim_ss"]].to_csv(
#     "data/chunk_v1.csv", index=False
# )

f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 18))
ax = ax.ravel()
_x = x[start:end]

ax[0].set_title(r"4th order model")
ax[0].step(
    # _x, p_up + p_low, label="kW ON/OFF", color="black", linestyle="--", alpha=0.3
    _x,
    p_up_data[start:end] + p_low_data[start:end],
    label="kW ON/OFF",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[0].step(_x, p_up + p_low, label="kW", color="green", linewidth=2)
ax[0].legend(loc="best")
ax[0].set_ylabel("kW")
ax[0].set_ylim([0, None])
ax[0].set_title("Power consumption")

ax[1].plot(
    _x,
    twu_data[start:end],
    label=r"T^{wu} data",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[1].plot(
    _x,
    setpoint_uz[start:end],
    label="Setpoint",
    linewidth=3,
    alpha=0.3,
    color="black",
)
# ax[1].plot(_x, Twu_sim, label="Twu", alpha=1.0, color="orange")
ax[1].plot(_x, Twu_sim, label=r"T^{wu}", color="green", linewidth=2)
ax[1].legend(loc="best")
ax[1].set_ylabel(r"[$^\circ$C]")
ax[1].set_title("Upper zone")

ax[2].plot(
    _x,
    twl_data[start:end],
    label=r"T^{wl} data",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[2].plot(
    _x,
    setpoint_lz[start:end],
    label="Setpoint",
    linewidth=3,
    alpha=0.3,
    color="black",
)
# ax[2].plot(_x, Twl_sim, label="Twl", alpha=1.0, color="orange")
ax[2].plot(_x, Twl_sim, label=r"T^{wl}", color="green", linewidth=2)
ax[2].legend(loc="best")
ax[2].set_ylabel(r"[$^\circ$C]")
ax[2].set_title("Lower zone")
ax[2].xaxis.set_tick_params(rotation=45)

_set_font_size(ax, misc=22,legend=20)
plt.tight_layout()
plt.savefig(BASE_FOLDER + "4thOrderModelVisualization.png", dpi=300)

f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 18))
ax = ax.ravel()
_x = x[start:end]

ax[0].set_title("Power consumption")
ax[0].step(
    # _x, p_up + p_low, label="kW ON/OFF", color="black", linestyle="--", alpha=0.3
    _x,
    p_up_data[start:end] + p_low_data[start:end],
    label=r"$P^{Base}$ ON/OFF",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[0].step(_x, p_up_ss + p_low_ss, label=r"$P^{Base}$", color="green", linewidth=2)
ax[0].legend(loc="best")
ax[0].set_ylabel("kW")
ax[0].set_ylim([0, None])

ax[1].plot(
    _x,
    twu_data[start:end],
    label=r"$T^{wu}$ ON/OFF",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[1].plot(
    _x,
    setpoint_uz[start:end],
    label="Setpoint",
    linewidth=3,
    alpha=0.3,
    color="black",
)
# ax[1].plot(_x, Twu_sim, label="Twu", alpha=1.0, color="orange")
ax[1].plot(_x, Twu_sim_ss, label=r"$T^{wu, Base}$", color="green", linewidth=2)
ax[1].legend(loc="best")
ax[1].set_ylabel(r"[$^\circ$C]")
ax[1].set_title("Upper zone")
ax[1].set_yticks([])

ax[2].plot(
    _x,
    twl_data[start:end],
    label=r"$T^{wl}$ ON/OFF",
    color="black",
    linestyle="--",
    alpha=0.3,
)
ax[2].plot(
    _x,
    setpoint_lz[start:end],
    label="Setpoint",
    linewidth=3,
    alpha=0.3,
    color="black",
)
# ax[2].plot(_x, Twl_sim, label="Twl", alpha=1.0, color="orange")
ax[2].plot(_x, Twl_sim_ss, label=r"$T^{wl, Base}$", color="green", linewidth=2)
ax[2].legend(loc="best")
ax[2].set_ylabel(r"[$^\circ$C]")
ax[2].xaxis.set_tick_params(rotation=45)
ax[2].set_title("Lower zone")
ax[2].set_yticks([])

_set_font_size(ax, misc=22,legend=20)
plt.tight_layout()
plt.savefig(BASE_FOLDER + "4thOrderModelVisualizationSteadyState.png", dpi=300)

plt.show()
