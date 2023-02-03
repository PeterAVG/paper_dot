#%% # noqa
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.base import Case
from src.common.utils import _set_font_size
from src.experiment_manager.cache import load_cache
from src.prepare_problem_v2 import get_scenarios_fcr

# matplotlib.pyplot.ion()

sns.set_theme()
sns.set(font_scale=1.5)


def plot_mFRR_case_result() -> None:
    cache = load_cache()

    params = {
        "elafgift": 0.0,
        "moms": 0.0,
        "year": 2022,
        "delta_max": 50,
        "one_lambda": False,
        "analysis": "analysis1",
        "run_oos": True,
        "case": Case.mFRR_AND_ENERGY.name,
    }

    information_results, _ = cache[params.__repr__()]
    # find index of result with most up_regulation_event in information_results
    index = np.argmax(
        [np.sum(information.up_regulation_event) for information in information_results]
    )
    information = information_results[index]

    fig, ax = plt.subplots(4, 1, figsize=(11, 16))
    ax = ax.ravel()
    x = np.array(list(range(24)))
    y = information.p_base - information.p_up_reserve
    ax[0].step(x, information.p_base, color="black", where="post")
    ax[0].step(x, y, color="black", linestyle="--", where="post")
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend([r"$P^{B}_{h}$", r"$P^{B}_{h} - p^{r,\uparrow}_{h}$"], loc="best")

    # Temperature dynamics
    w = -1
    # _w = information.lambda_rp.shape[0]
    x2 = np.array(list(range(24 * 60))) / 60
    # ax[1].set_title("Scenario {}".format(w))
    ax[1].plot(x2, information.twu[w, :], label=r"$T^{wu}_{t}$")
    ax[1].plot(x2, information.twl[w, :], label=r"$T^{wl}_{t}$")
    ax[1].plot(x2, information.twu_base, label=r"$T^{wu, Base}_{t}$")
    ax[1].plot(x2, information.twl_base, label=r"$T^{wl, Base}_{t}$")
    # ax[1].plot(x2, information.twu_data, label="Twu-meas", color="black", linestyle=":")
    # ax[1].plot(x2, information.twl_data, label="Twl-meas", color="black", linestyle=":")
    # ax[1].plot(x2, t_f[w, :], label="TcFood")
    ax[1].set_ylabel(r"Temperature [$^\circ$C]")
    ax[1].legend(loc="best")
    # ax[1].set_xlabel("Time [h]")

    # power dynamics
    _pt = information.pt[w, :]

    ax[2].step(x, information.p_base, label=r"$P^{B}_{h}$", color="black", where="post")
    ax[2].step(
        x,
        _pt,
        label=r"$P^{B}_{h} - p^{b,\uparrow}_{h} + p^{b,\downarrow}_{h}$",
        color="black",
        linestyle="--",
        where="post",
    )
    # ax[2].set_xlabel("Time [h]")
    ax[2].set_ylabel("Power [kW]")
    # ax[2].set_title("Scenario {}".format(w))
    ax[2].legend(loc="best")

    ax[3].step(
        x,
        information.lambda_spot[w, :],
        label=r"$\lambda_{h}^{s}$",
        color="red",
        where="post",
    )

    ax[3].step(
        x,
        information.lambda_rp[w, :],
        label=r"$\lambda_{h}^{b}$",
        color="blue",
        where="post",
    )
    ax[3].step(
        x,
        information.lambda_spot[w, :] + information.lambda_b[w, :],
        label=r"$\lambda_{h}^{bid}$",
        color="orange",
        where="post",
    )
    ax[3].legend(loc="best")
    ax[3].set_ylabel("Price [DKK/kWh]")
    ax[3].set_xlabel("Time [h]")
    # plt.rcParams.update({"font.size": 20})
    _set_font_size(ax, legend=16)
    plt.tight_layout()

    plt.savefig("tex/figures/mFRR_single_case", dpi=300)


def plot_fcr_case_result() -> None:
    cache = load_cache()

    params = {
        "elafgift": 0.0,
        "moms": 0.0,
        "year": 2022,
        "delta_max": 5,
        "analysis": "analysis1",
        "run_oos": True,
        "case": Case.FCR.name,
    }

    information_results, _ = cache[params.__repr__()]
    # find index of result with most freuquenc response in information_results
    index = np.argmax(
        [np.sum(abs(information.frequency)) for information in information_results]
    )
    index = np.argmax(
        [
            np.sum(
                (information.twu_base - information.twu)
                + (information.twl_base - information.twl)
            )
            for information in information_results
        ]
    )
    information = information_results[index]

    fig, ax = plt.subplots(4, 1, figsize=(11, 16))
    ax = ax.ravel()
    x = np.array(list(range(24)))
    y = information.p_base - information.p_up_reserve
    ax[0].step(x, information.p_base, color="black", where="post")
    ax[0].step(x, y, color="black", linestyle="--", where="post")
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend([r"$P^{B}_{h}$", r"$P^{B}_{h} - p^{cap}$"], loc="best")
    ax[0].set_xlim([0, 24])
    # Temperature dynamics
    w = -1
    # _w = information.lambda_rp.shape[0]
    x2 = np.array(list(range(24 * 60))) / 60
    # ax[1].set_title("Scenario {}".format(w))
    ax[1].plot(x2, information.twu[w, :], label=r"$T^{wu}_{t}$")
    ax[1].plot(x2, information.twl[w, :], label=r"$T^{wl}_{t}$")
    ax[1].plot(x2, information.twu_base, label=r"$T^{wu, Base}_{t}$", alpha=0.3)
    ax[1].plot(x2, information.twl_base, label=r"$T^{wl, Base}_{t}$", alpha=0.3)
    # ax[1].plot(x2, information.twu_data, label="Twu-meas", color="black", linestyle=":")
    # ax[1].plot(x2, information.twl_data, label="Twl-meas", color="black", linestyle=":")
    # ax[1].plot(x2, t_f[w, :], label="TcFood")
    ax[1].set_ylabel(r"Temperature [$^\circ$C]")
    ax[1].legend(loc="best")
    # ax[1].set_xlabel("Time [h]")
    ax[1].set_xlim([0, 24])

    # power dynamics
    _pt = information.pt[w, :]
    _pt_lz = information.pt_lz[w, :]
    _pt_uz = information.pt_uz[w, :]
    x2 = np.array(list(range(24 * 60))) / 60

    ax[2].step(
        x2,
        np.repeat(information.p_base, 60),
        # label=r"$P^{B}_{h}$",
        color="black",
        where="post",
    )
    ax[2].step(
        x2,
        _pt,
        label=r"$P_{h}$",
        color="black",
        linestyle="--",
        where="post",
    )
    ax[2].step(
        x2,
        np.repeat(information.p_base_lz, 60),
        # label=r"$P^{B,lz}_{h}$",
        color="red",
        where="post",
        alpha=0.3,
    )
    ax[2].step(
        x2,
        _pt_lz,
        label=r"$P^{lz}_{h}$",
        color="orange",
        linestyle="--",
        where="post",
        alpha=0.3,
    )
    ax[2].step(
        x2,
        np.repeat(information.p_base_uz, 60),
        # label=r"$P^{B,uz}_{h}$",
        color="green",
        where="post",
        alpha=0.3,
    )
    ax[2].step(
        x2,
        _pt_uz,
        label=r"$P^{uz}_{h}$",
        color="blue",
        linestyle="--",
        where="post",
        alpha=0.3,
    )
    # ax[2].set_xlabel("Time [h]")
    ax[2].set_ylabel("Power [kW]")
    # ax[2].set_title("Scenario {}".format(w))
    ax[2].legend(loc="upper right")
    ax[2].set_xlim([0, 24])

    ax[3].step(
        x,
        information.lambda_spot[w, :],
        label=r"$\lambda_{h}^{s}$",
        color="red",
        where="post",
    )

    ax[3].step(
        x,
        information.lambda_fcr[w, :],
        label=r"$\lambda_{h}^{FCR}$",
        color="blue",
        where="post",
    )
    ax[3].legend(loc="best")
    ax[3].set_ylabel("Price [DKK/kWh]")
    ax[3].set_xlabel("Time [h]")
    ax[3].set_xlim([0, 24])
    # plt.rcParams.update({"font.size": 20})
    _set_font_size(ax, legend=16)
    plt.tight_layout()

    plt.savefig("tex/figures/fcr_single_case", dpi=300)


def plot_fcr_prices() -> None:

    scenarios = get_scenarios_fcr(1, True, 2022)
    fcr_prices = scenarios.lambda_fcr.reshape(-1)  # type:ignore
    spot_prices = scenarios.lambda_spot.reshape(-1)  # type:ignore

    # date where FCR market opens up to continental Europe
    cutoff_idx = (246 + 1) * 24
    # print desciptive statistics of fcr prices after cutoff
    print(pd.Series(fcr_prices[cutoff_idx:]).describe())

    _hours = list(range(1, len(fcr_prices) + 1))
    # convert list of integers 'days' to datetimes:
    hours = [
        datetime.datetime(2022, 1, 1) + datetime.timedelta(hours=h)  # type:ignore
        for h in _hours
    ]

    # NOTE: base temperature have jumps due to aggregation of p_base to hourly resolution
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # lineplots of prices
    ax.step(
        hours,
        spot_prices,
        label=r"$\lambda_{h}^{s}$",
        color="red",
        where="post",
        alpha=0.5,
    )
    ax.step(
        hours,
        fcr_prices,
        label=r"$\lambda_{h}^{FCR}$",
        color="blue",
        where="post",
    )
    # vertical line at cutoff_idx
    ax.axvline(hours[cutoff_idx], color="black", linestyle="--", alpha=0.5)

    ax.set_ylabel("Price [DKK/kWh]")
    ax.legend(loc="best")
    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    _set_font_size(ax, legend=16)

    plt.savefig("tex/figures/fcr_prices", dpi=300)


def plot_yearly_earnings() -> None:
    cache = load_cache()

    params = {
        "elafgift": 0.0,
        "moms": 0.0,
        "year": 2022,
        "delta_max": 5,
        "analysis": "analysis1",
        "run_oos": True,
        "case": Case.FCR.name,
    }
    _, fcr_results = cache[params.__repr__()]

    params = {
        "elafgift": 0.0,
        "moms": 0.0,
        "year": 2022,
        "delta_max": 50,
        "one_lambda": False,
        "analysis": "analysis1",
        "run_oos": True,
        "case": Case.mFRR_AND_ENERGY.name,
    }
    _, mfrr_results = cache[params.__repr__()]

    print(len(fcr_results))
    print(len(mfrr_results))
    assert len(fcr_results) == len(mfrr_results)

    assert all(
        [
            np.isclose(e1.base_cost_today, e2.base_cost_today)
            for e1, e2 in zip(fcr_results, mfrr_results)
        ]
    )
    base_cost_today = [e.base_cost_today for e in fcr_results]
    fcr_cost = [e.total_cost for e in fcr_results]
    mfrr_cost = [e.total_cost for e in mfrr_results]

    assert len(fcr_cost) == len(mfrr_cost) == len(base_cost_today)

    days = list(range(1, len(base_cost_today) + 1))
    # convert list of integers 'days' to datetimes:
    days = [
        datetime.datetime(2022, 1, 1) + datetime.timedelta(days=d)  # type:ignore
        for d in days
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(
        days,
        np.cumsum(base_cost_today),
        label="Base cost",
        color="black",
        linestyle="--",
    )
    ax.plot(days, np.cumsum(fcr_cost), label="FCR")
    ax.plot(days, np.cumsum(mfrr_cost), label="mFRR")
    ax.set_ylabel("Cumulative cost [DKK]")
    ax.legend(loc="best")
    ax.xaxis.set_tick_params(rotation=45)
    _set_font_size(ax, legend=16)
    plt.tight_layout()
    plt.savefig("tex/figures/cumulative_cost_comparison.png", dpi=300)
    # plt.show()


def plot_profit_vs_delta_max() -> None:

    cache = load_cache()

    all_fcr_results = [
        (eval(k)["delta_max"], v)
        for k, v in cache.items()
        if (eval(k)["case"] == Case.FCR.name and eval(k)["delta_max"] <= 10)
    ]
    print(len(all_fcr_results))

    delta_max = [e[0] for e in all_fcr_results]
    fcr_cost = [[sum(q.total_cost for q in e[1][1])] for e in all_fcr_results]
    base_cost = [[sum(q.base_cost_today for q in e[1][1])] for e in all_fcr_results]
    idx_sorted = np.argsort(delta_max)

    for d, f, b in zip(delta_max, fcr_cost, base_cost):
        print(f"DeltaT: {d} & FCR: {round(f[0])} & Base: {round(b[0])}")

    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    # make lineplot with delta_max vs. profit and "X" as markers
    ax.plot(
        np.array(delta_max)[idx_sorted],
        np.array(base_cost)[idx_sorted] - np.array(fcr_cost)[idx_sorted],
        marker="X",
        markersize=10,
    )
    # scientific notation on y-axis
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    ax.set_xlabel(r"Temperature deviation [$^\circ$C]")
    ax.set_ylabel("Savings [DKK]")
    # ax.legend(loc="best")
    ax.xaxis.set_tick_params(rotation=45)
    _set_font_size(ax, legend=16)

    plt.tight_layout()
    plt.savefig("tex/figures/profit_vs_delta_temp.png", dpi=300)
    plt.show()


def main() -> None:
    if True:
        plot_mFRR_case_result()
        plot_fcr_case_result()
        plot_yearly_earnings()
        plot_fcr_prices()
        plot_profit_vs_delta_max()


if __name__ == "__main__":
    main()
