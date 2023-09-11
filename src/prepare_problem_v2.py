from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyomo.environ import AbstractModel, Objective, maximize

from src.base import SCENARIO_PATH, Case
from src.objective import (
    o_expected_activation_payment,
    o_expected_energy_consumption,
    o_expected_energy_cost,
    o_expected_fcr_reserve_payment,
    o_expected_rebound_cost,
    o_expected_reserve_payment,
    o_fcr_penalty,
    o_penalty,
    o_rule,
    o_rule_fcr_energy,
)
from src.problem_v2 import OptimizationInstance, SolverInstance, value


@dataclass
class OptimizationResult:
    base_cost_today: float
    total_cost: float
    expected_energy_cost: float
    rebound_cost: float
    reserve_payment: float
    act_payment: float
    penalty_cost: float
    battery_capacity: float

    def __repr__(self) -> str:
        # print and round to 3 decimals:
        reserve_payment = f"reserve_payment {round(self.reserve_payment, 3)}"
        act_payment = f"activation_payment {round(self.act_payment, 3)}"
        expected_energy_cost = (
            f"expected_energy_cost {round(self.expected_energy_cost, 3)}"
        )
        rebound_cost = f"rebound_cost {round(self.rebound_cost, 3)}"
        total_cost = f"total_cost {round(self.total_cost, 3)}"
        base_cost_today = f"base_cost_today {round(self.base_cost_today, 3)}"
        penalty_cost = f"penalty_cost {round(self.penalty_cost, 3)}"
        battery_capacity = f"battery_capacity {round(self.battery_capacity, 3)}"
        return f"{reserve_payment}\n {act_payment}\n {expected_energy_cost}\n {rebound_cost}\n {total_cost}\n {base_cost_today}\n {penalty_cost}\n {battery_capacity}"


@dataclass
class Scenario:
    up_regulation_event: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    lambda_mfrr: np.ndarray
    prob: np.ndarray
    frequency: Optional[np.ndarray] = None
    lambda_fcr: Optional[np.ndarray] = None


@dataclass
class InstanceInformation:
    p_up_reserve: np.ndarray
    p_base: np.ndarray
    p_base_lz: np.ndarray
    p_base_uz: np.ndarray
    pt: np.ndarray
    pt_lz: np.ndarray
    pt_uz: np.ndarray
    p_up: np.ndarray
    p_down: np.ndarray
    s: np.ndarray
    slack: np.ndarray
    p_up_lz: np.ndarray
    u_up_lz: np.ndarray
    y_up_lz: np.ndarray
    z_up_lz: np.ndarray
    z_down_lz: np.ndarray
    y_down_lz: np.ndarray
    u_down_lz: np.ndarray
    p_down_lz: np.ndarray
    p_up_uz: np.ndarray
    u_up_uz: np.ndarray
    y_up_uz: np.ndarray
    z_up_uz: np.ndarray
    z_down_uz: np.ndarray
    y_down_uz: np.ndarray
    u_down_uz: np.ndarray
    p_down_uz: np.ndarray

    up_regulation_event: np.ndarray
    g_indicator: np.ndarray
    lambda_b: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray

    # TCL specific temperature data
    tzl_base: np.ndarray
    tzl: np.ndarray
    twl_base: np.ndarray
    twl: np.ndarray
    tzu_base: np.ndarray
    tzu: np.ndarray
    twu_base: np.ndarray
    twu: np.ndarray

    twu_data: np.ndarray
    twl_data: np.ndarray

    alpha: float
    beta: float


@dataclass
class FCRInstanceInformation:
    p_up_reserve: np.ndarray
    p_up_reserve_lz: np.ndarray
    p_up_reserve_uz: np.ndarray
    p_base: np.ndarray
    p_base_lz: np.ndarray
    p_base_uz: np.ndarray
    pt: np.ndarray
    pt_lz: np.ndarray
    pt_uz: np.ndarray
    p_freq: np.ndarray
    p_freq_lz: np.ndarray
    p_freq_uz: np.ndarray
    s: np.ndarray
    s_lz: np.ndarray
    s_uz: np.ndarray
    s_abs: np.ndarray
    s_abs_lz: np.ndarray
    s_abs_uz: np.ndarray
    frequency: np.ndarray
    lambda_spot: np.ndarray
    lambda_fcr: np.ndarray
    tzl_base: np.ndarray
    tzl: np.ndarray
    twl_base: np.ndarray
    twl: np.ndarray
    tzu_base: np.ndarray
    tzu: np.ndarray
    twu_base: np.ndarray
    twu: np.ndarray
    twu_data: np.ndarray
    twl_data: np.ndarray


def build_uncertainty_set(
    scenarios: np.ndarray, nb: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(2021)
    # scenario_set = defaultdict(int)
    # for w in range(scenarios.shape[0]):
    #     s = scenarios[w,:]
    #     s = tuple(s.tolist())
    #     scenario_set[s] += 1

    # print(len(scenario_set))
    # sorted(scenario_set.items(), key=lambda x: x[1], reverse=True)
    # dict(sorted(scenario_set.items(), key=lambda x: x[1], reverse=True)).values()

    weights = (
        pd.Series(scenarios.sum(axis=1))
        .sort_values(ascending=True)
        .to_frame()
        .rename(columns={0: "count"})
    )
    weights = weights.groupby("count").size()
    probs = (weights / weights.sum()).values

    # sample scenarios according to sum of daily activation hours
    new_scenarios = np.empty(shape=(len(probs), scenarios.shape[1]))
    for j, h in enumerate(weights.index.values):
        idx = scenarios.sum(axis=1) == h
        possible_set = scenarios[idx, :]
        # get random choice
        i = rng.choice(list(range(possible_set.shape[0])), size=1)[0]
        new_scenarios[j, :] = possible_set[i, :]

    assert len(probs) == new_scenarios.shape[0]

    if nb is not None:
        assert new_scenarios.shape[0] % nb == 0 and nb < 20, (
            nb,
            new_scenarios.shape[0],
        )
        i = new_scenarios.shape[0] // nb
        new_scenarios = new_scenarios[::i, :]
        probs = probs[::i]
        probs /= probs.sum()  # type:ignore
        assert nb == new_scenarios.shape[0], (nb, new_scenarios.shape[0])

    assert len(probs) == new_scenarios.shape[0]
    assert np.isclose(1, probs.sum())

    return new_scenarios, probs


def find_rp_price(x: pd.Series) -> float:
    if x.flag == 1 and x.flag_down == 0:
        return x.BalancingPowerPriceUpDKK
    elif x.flag_down == 1 and x.flag == 0:
        return x.BalancingPowerPriceDownDKK
    elif x.flag_down == 0 and x.flag == 0:
        return x.SpotPriceDKK
    elif x.flag_down == 1 and x.flag == 1:
        if (x.SpotPriceDKK - x.BalancingPowerPriceDownDKK) > (
            x.BalancingPowerPriceUpDKK - x.SpotPriceDKK
        ):
            return x.BalancingPowerPriceDownDKK
        else:
            return x.BalancingPowerPriceUpDKK
    else:
        raise Exception


def build_uncertainty_set_v2(
    df_scenarios: pd.DataFrame, nb: int = 1, run_oos: bool = False
) -> Scenario:

    assert nb >= 1
    rng = np.random.default_rng(4)

    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")

    weights = df_scenarios.groupby(df_scenarios.Date).flag.sum().value_counts()
    # probs = (weights / weights.sum()).values

    # sample scenarios according to sum of daily activation hours
    up_regulation_event = np.empty(shape=(len(weights) * nb, 24))
    prob = np.empty(len(weights) * nb)
    lambda_rp = np.empty(shape=(len(weights) * nb, 24))
    lambda_mfrr = np.empty(shape=(len(weights) * nb, 24))
    lambda_spot = np.empty(shape=(len(weights) * nb, 24))

    # overwrite lambda spot- and mFRR with average prices
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values
    df_scenarios["lambda_rp_diff"] = (
        df_scenarios["lambda_rp"] - df_scenarios["SpotPriceDKK"]
    )
    agg = df_scenarios.groupby("Hour").mean()
    lambda_spot[:] = np.repeat(
        agg.SpotPriceDKK.values.reshape(1, -1), len(weights) * nb, axis=0
    )
    lambda_mfrr[:] = np.repeat(
        agg.mFRR_UpPriceDKK.values.reshape(1, -1), len(weights) * nb, axis=0
    )

    # create list of all rp-up - spot prices when up-regulation happened.
    lambda_rp_up_diff = df_scenarios.query(
        "flag == 1 & flag_down == 0"
    ).lambda_rp_diff.values
    # create list of all rp - spot prices when up-regulation didn't happened.
    lambda_rp_diff = df_scenarios.query("flag == 0").lambda_rp_diff.values

    count = 0
    for j, w in weights.to_dict().items():

        dates = (
            df_scenarios.groupby(df_scenarios.Date)
            .flag.sum()
            .to_frame()
            .query(f"flag == {j}")
            .index.values.tolist()
        )
        assert len(dates) >= 1

        if j > 0:
            _dates = rng.choice(dates, replace=False, size=nb).tolist()
        else:
            # for j==0, no up-regulation happen and all days are the same
            _dates = rng.choice(dates, replace=False, size=1).tolist()

        assert len(_dates) >= 1

        for n in range(len(_dates)):

            # sample historical day where up-regulation happened
            date = _dates[n]  # noqa
            sample = df_scenarios.query("Date == @date")
            up_regulation_event[count, :] = sample.flag.values
            ix = np.where(sample.flag.values == 1)[0]

            # sample up-reg prices when up-regulation happened and rp-reg when it didn't
            lambda_rp_up_diff_sample = rng.choice(lambda_rp_up_diff, size=j)
            lambda_rp_diff_sample = rng.choice(lambda_rp_diff, size=24)

            # lambda_spot[count, :] = sample.SpotPriceDKK.values

            lambda_rp[count, :] = lambda_spot[0, :] + lambda_rp_diff_sample
            lambda_rp[count, ix] = lambda_spot[0, ix] + lambda_rp_up_diff_sample

            prob[count] = w / len(_dates)

            count += 1

    prob = prob[:count]
    prob /= sum(prob)

    if run_oos:
        pass
    else:
        # kind of robust approach..
        prob = np.ones(count) / count

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    return Scenario(
        up_regulation_event=up_regulation_event[:count, :],
        lambda_rp=lambda_rp[:count, :] / 1000,
        lambda_spot=lambda_spot[:count, :] / 1000,
        lambda_mfrr=lambda_mfrr[:count, :] / 1000,
        prob=prob,
    )


def build_test_uncertainty_set(df_scenarios: pd.DataFrame) -> Scenario:
    rng = np.random.default_rng(4)

    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")

    # choose 2 random days from the dataset
    dates = df_scenarios.Date.unique()
    dates = rng.choice(dates, size=2, replace=False).tolist()
    lambda_spot = df_scenarios.query("Date == @dates").SpotPriceDKK.values.reshape(
        2, -1
    )
    lambda_spot = np.tile(lambda_spot, (10, 1))
    lambda_rp = (
        lambda_spot
        + rng.normal(0, 300, size=lambda_spot.shape)
        + rng.integers(0, 2, size=lambda_spot.shape) * 400
    )
    lambda_rp = np.maximum(0, lambda_rp)

    # for i in range(1, lambda_rp.shape[0]):
    #     lambda_rp[i, :] = lambda_rp[0, :]

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)
    lambda_mfrr = (
        df_scenarios.groupby("Hour").mFRR_UpPriceDKK.mean().values.reshape(1, -1)
    )
    lambda_mfrr = np.tile(lambda_mfrr, (20, 1))

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(20) / 20,
    )


def build_oos_scenarios(df_scenarios: pd.DataFrame) -> Scenario:
    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values

    lambda_spot = df_scenarios.SpotPriceDKK.values.reshape(-1, 24)
    lambda_rp = df_scenarios.lambda_rp.values.reshape(-1, 24)
    lambda_mfrr = df_scenarios.mFRR_UpPriceDKK.values.reshape(-1, 24)

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(lambda_rp.shape[0]) / lambda_rp.shape[0],
    )


def get_arbitrary_scenarios(
    df_scenarios: pd.DataFrame, nb_spot: int = 10, nb_rp: int = 1, seed: int = 1
) -> Scenario:
    rng = np.random.default_rng(seed)
    # TODO: use DK1 prices!!!
    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values
    df_scenarios["lambda_rp_diff"] = (
        df_scenarios["lambda_rp"] - df_scenarios["SpotPriceDKK"]
    )

    # choose 'nb_spot' random days from the dataset for spot prices
    dates = df_scenarios.Date.unique()
    _dates = rng.choice(dates, size=nb_spot, replace=False).tolist()  # noqa
    lambda_spot = df_scenarios.query("Date == @_dates").SpotPriceDKK.values.reshape(
        nb_spot, 24
    )
    lambda_spot = np.tile(lambda_spot, (nb_rp, 1))
    lambda_spot = lambda_spot.round()

    # distribution of balancing price differentials
    # TODO: use a better distribution
    assert nb_spot * nb_rp <= 365
    dates = (
        df_scenarios.groupby(df_scenarios.Date)
        .flag.sum()
        .sort_values()
        .drop_duplicates()
        .index.values.tolist()
    )
    # weights = df_scenarios.groupby(df_scenarios.Date).flag.sum().value_counts()
    replace = True if nb_spot > len(dates) else False
    _dates = rng.choice(dates, size=nb_spot, replace=replace).tolist()  # noqa
    lambda_rp_diff_set = np.array(
        [
            df_scenarios.query(f"Date == '{_date}'").lambda_rp_diff.values
            for _date in _dates
        ]
    )
    lambda_rp_diff_set = lambda_rp_diff_set.reshape(-1, 24)
    # lambda_rp_diff_set = df_scenarios.query(
    #     "Date == @_dates"
    # ).lambda_rp_diff.values.reshape(-1, 24)
    # replace = False if nb_spot * nb_rp * 24 <= lambda_rp_diff_set.shape[0] else True
    # lambda_rp_diff = rng.choice(
    #     lambda_rp_diff_set, size=nb_spot * nb_rp * 24, replace=replace
    # )  # noqa
    # lambda_rp_diff = lambda_rp_diff.reshape(nb_spot * nb_rp, 24)
    # lambda_rp_diff = np.tile(lambda_rp_diff_set, (nb_spot, 1))
    lambda_rp = lambda_spot + lambda_rp_diff_set
    lambda_rp = lambda_rp.round()

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)
    lambda_mfrr = (
        df_scenarios.groupby("Hour").mFRR_UpPriceDKK.mean().values.reshape(1, -1)
    )
    lambda_mfrr = np.tile(lambda_mfrr, (nb_spot, 1))

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(nb_spot * nb_rp) / (nb_spot * nb_rp),
    )


def get_scenarios_fcr(
    nb_spot: int, run_oos: bool, year: int, seed: int = 1
) -> Scenario:
    rng = np.random.default_rng(seed)

    # prices
    date_cols = ["DATE_FROM", "DATE_TO"]
    df1 = pd.read_csv("data/fcr_2021.csv", parse_dates=date_cols)
    df2 = pd.read_csv("data/fcr_2022.csv", parse_dates=date_cols)
    df3 = pd.read_csv("data/fcr_2023.csv", parse_dates=date_cols)

    dk1 = ["DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
    dk2 = ["DENMARK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]

    df1 = df1[date_cols + dk1]
    df2 = df2[date_cols + dk2]
    df3 = df3[date_cols + dk2]

    df1.rename(columns={dk1[0]: "DK_PRICE"}, inplace=True)
    df2.rename(columns={dk2[0]: "DK_PRICE"}, inplace=True)
    df3.rename(columns={dk2[0]: "DK_PRICE"}, inplace=True)

    df = pd.concat([df1, df2, df3]).sort_values(by=date_cols).reset_index(drop=True)

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
    df2["DateOnly"] = pd.to_datetime(df2.Date.dt.date)
    df2 = df2.query(f"DateOnly.dt.year == {year}")

    # find corresponding spot prices
    # TODO: use DK1
    spot = pd.read_csv(
        SCENARIO_PATH,
        parse_dates=["HourUTC"],
    ).query(f"HourUTC.dt.year == {year}")
    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        spot.groupby(spot.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    spot = spot.query("Date != @dates_to_substract")
    # spot = spot.query("Date <= '2022-01-10'")

    # merge spot with df2 on Date and HourUTC and keep only what is in spot
    df2 = df2.merge(
        spot[["HourUTC", "SpotPriceDKK"]],
        left_on=["Date"],
        right_on=["HourUTC"],
        how="inner",
    )

    # grid frequency
    min1 = pd.read_csv("data/germany_merger_1min_v2.csv")
    # min1 = pd.read_csv("data/germany_merger_1min_v2.csv", parse_dates=["datetime"])
    # min1.sort_values(by="datetime", inplace=True)
    # min1.drop_duplicates("datetime", inplace=True)
    min1.Hz = (min1.Hz.values - 50) * 1000
    print(min1.shape)

    # sample "nb_spot" dates from df2 columns "Date"
    if not run_oos:
        sampled_dates = rng.choice(  # noqa
            df2.DateOnly.tolist(), size=nb_spot, replace=False
        ).tolist()
        sampled_fcr = df2.query("DateOnly in @sampled_dates").DK_PRICE.values
        assert sampled_fcr.shape[0] == nb_spot * 24
        sampled_spot = df2.query("DateOnly in @sampled_dates").SpotPriceDKK.values
        assert sampled_spot.shape[0] == nb_spot * 24
    else:
        sampled_fcr = df2.DK_PRICE.values
        assert sampled_fcr.shape[0] % 24 == 0
        sampled_spot = df2.SpotPriceDKK.values
        nb_spot = sampled_fcr.shape[0] // 24
        assert sampled_spot.shape[0] % 24 == 0
        # _dates = df2.DateOnly.dt.date.tolist()
        # frequency = min1.query("datetime.dt.date in @_dates").Hz.values
        # sampled_spot.shape[0] * 60 - frequency.shape[0]

    # sample "nb_spot" times in min1 where each sample has length 60*24*nb_spot and are non-overlapping
    freq = min1.Hz.tolist()
    sampled_freq_list = []
    for i in range(nb_spot):
        start = rng.choice(range(len(freq) - 60 * 24), size=1)[0]
        sampled_freq_list.append(freq[start : start + 60 * 24])
        freq = freq[:start] + freq[start + 60 * 24 :]
        if len(freq) <= 60 * 24 * 2:
            freq = min1.Hz.tolist()

    assert len(sampled_freq_list) == nb_spot
    assert len(np.concatenate(sampled_freq_list)) / 60 == len(sampled_fcr)
    assert sampled_fcr.shape == sampled_spot.shape

    sampled_freq = np.concatenate(sampled_freq_list)

    # clip frequency to a response between -1 and 1: 1 when frequency is below -200 mHz and -1 when frequency is above 200 mHz
    # there is a deadband between [-20 mHz: 20 mHz]
    ix1 = sampled_freq >= 200
    ix2 = sampled_freq <= -200
    ix3 = (sampled_freq > -20) & (sampled_freq < 20)
    ix4 = (sampled_freq <= -20) & (sampled_freq > -200)
    ix5 = (sampled_freq >= 20) & (sampled_freq < 200)
    sampled_freq[ix3] = 0.0
    sampled_freq[ix4] = (sampled_freq[ix4] + 20) / (200 - 20)
    sampled_freq[ix5] = (sampled_freq[ix5] - 20) / (200 - 20)
    sampled_freq[ix1] = 1
    sampled_freq[ix2] = -1
    assert sampled_freq.min() >= -1
    assert sampled_freq.max() <= 1
    assert all(sampled_freq[(sampled_freq <= -20) & (sampled_freq > -200)] <= 0)
    assert all(sampled_freq[(sampled_freq >= 20) & (sampled_freq < 200)] >= 0)
    assert any(sampled_freq != 0)
    assert (sampled_freq < -1).sum() == 0
    assert (sampled_freq > 1).sum() == 0

    print(sampled_freq.min())
    print(sampled_freq.max())

    lambda_fcr = sampled_fcr.reshape(-1, 24) / 1000
    lambda_spot = sampled_spot.reshape(-1, 24) / 1000
    frequency = sampled_freq.reshape(-1, 24 * 60)
    lambda_mfrr = sampled_spot.reshape(-1, 24) / 1000
    lambda_rp = sampled_spot.reshape(-1, 24) / 1000
    prob = np.ones(frequency.shape[0]) / frequency.shape[0]

    return Scenario(
        lambda_fcr=lambda_fcr,
        lambda_spot=lambda_spot,
        frequency=frequency,
        lambda_mfrr=lambda_mfrr,
        lambda_rp=lambda_rp,
        prob=prob,
        up_regulation_event=np.ones(shape=lambda_fcr.shape),
    )


def get_scenarios_from_lookback(df: pd.DataFrame, lookback: int) -> Scenario:
    start_date = datetime(2022, 1, 1) - timedelta(days=lookback)
    df_scenarios = df.query(f"HourUTC >= '{start_date}'")

    lambda_spot = df_scenarios["SpotPriceDKK"].values.reshape(-1, 24)
    lambda_mfrr = df_scenarios["mFRR_UpPriceDKK"].values.reshape(-1, 24)

    lambda_rp = df_scenarios.lambda_rp.values.reshape(-1, 24)
    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    assert lambda_spot.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape
    assert lambda_rp.shape == lambda_mfrr.shape

    scenarios = Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp,
        lambda_spot=lambda_spot,
        lambda_mfrr=lambda_mfrr,
        prob=np.ones(lambda_rp.shape[0]) / lambda_rp.shape[0],
    )

    return scenarios


def sample_lambda_rp(df: pd.DataFrame, size: int, seed: int) -> np.ndarray:
    # sample dates where up-regulation happend "flag.sum() times"
    dates = (
        df.groupby(df.Date)
        .flag.sum()
        .sort_values()
        .drop_duplicates()
        .index.values.tolist()
    )
    rng = np.random.default_rng(seed)
    sampled_dates = rng.choice(dates, size=size, replace=True).tolist()  # noqa
    lambda_rp_diff_set = np.array(
        [
            df.query(f"Date == '{_date}'").lambda_rp_diff.values
            for _date in sampled_dates
        ]
    ).reshape(-1, 24)

    return lambda_rp_diff_set


def get_chunk_instance(scenarios: Scenario, case: Case) -> OptimizationInstance:

    df = pd.read_csv("data/chunk_v1.csv")
    assert df.shape[0] % 60 == 0, "Data must be in 60 minute chunks"

    setpoint_lz = 447
    setpoint_uz = 455

    # twl_data = setpoint_lz * np.ones(df.shape[0])
    # twu_data = setpoint_uz * np.ones(df.shape[0])
    twl_data = df.Twl_sim_ss.values
    twu_data = df.Twu_sim_ss.values
    p_base_lz = np.mean(df.KW_LZ_SS.values.reshape(-1, 60), axis=1).round(1)
    p_base_uz = np.mean(df.KW_UZ_SS.values.reshape(-1, 60), axis=1).round(1)
    p_base = p_base_lz + p_base_uz
    regime = df.regime_UZ.values

    p_nom = 442  # kW
    p_min = 0
    n_steps = 24 * 60
    n_steps = df.shape[0]
    dt = 1 / 60
    ta = 20.0  # degrees Celsius

    # parameters of physical model
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

    lambda_fcr = scenarios.lambda_fcr if case.name == Case.FCR.name else None
    frequency = scenarios.frequency if case.name == Case.FCR.name else None

    return OptimizationInstance(
        lambda_mfrr=scenarios.lambda_mfrr,
        lambda_rp=scenarios.lambda_rp,
        lambda_spot=scenarios.lambda_spot,
        up_regulation_event=scenarios.up_regulation_event,
        probabilities=scenarios.prob,
        setpoint_lz=setpoint_lz,
        setpoint_uz=setpoint_uz,
        twl_data=twl_data,
        twu_data=twu_data,
        p_base_lz=p_base_lz,
        p_base_uz=p_base_uz,
        p_base=p_base,
        regime=regime,
        p_nom=p_nom,
        p_min=p_min,
        n_steps=n_steps,
        dt=dt,
        ta=ta,
        Czu=Czu,
        Czl=Czl,
        Cwu=Cwu,
        Cwl=Cwl,
        Rww=Rww,
        Rwz=Rwz,
        Rzuzl=Rzuzl,
        Rwua1=Rwua1,
        Rwua2=Rwua2,
        Rwla1=Rwla1,
        delta_max=150,
        max_up_time=5,
        min_up_time=1,
        elafgift=0.0,
        moms=0.0,
        tariff=np.zeros(24),
        one_lambda=True,
        M=15,
        max_lambda_bid=5,
        rebound=4,
        lambda_fcr=lambda_fcr,
        frequency=frequency,
    )


def get_variables_and_params(
    instance: AbstractModel, case: Case
) -> Union[InstanceInformation, FCRInstanceInformation]:

    if case.name == Case.mFRR_AND_ENERGY.name:
        # extract solver results
        p_up_reserve = np.array(list(instance.p_up_reserve.extract_values().values()))
        p_base = np.array(list(instance.p_base.extract_values().values()))
        pt = np.array(list(instance.pt.extract_values().values())).reshape(-1, 24)
        pt_lz = np.array(list(instance.pt_lz.extract_values().values())).reshape(-1, 24)
        pt_uz = np.array(list(instance.pt_uz.extract_values().values())).reshape(-1, 24)
        p_up = np.array(list(instance.p_up.extract_values().values())).reshape(-1, 24)
        p_down = np.array(list(instance.p_down.extract_values().values())).reshape(
            -1, 24
        )
        p_base_lz = np.array(list(instance.p_base_lz.extract_values().values()))
        p_base_uz = np.array(list(instance.p_base_uz.extract_values().values()))
        p_up_lz = np.array(list(instance.p_up_lz.extract_values().values())).reshape(
            -1, 24
        )
        u_up_lz = np.array(list(instance.u_up_lz.extract_values().values())).reshape(
            -1, 24
        )
        y_up_lz = np.array(list(instance.y_up_lz.extract_values().values())).reshape(
            -1, 24
        )
        z_up_lz = np.array(list(instance.z_up_lz.extract_values().values())).reshape(
            -1, 24
        )
        z_down_lz = np.array(
            list(instance.z_down_lz.extract_values().values())
        ).reshape(-1, 24)
        y_down_lz = np.array(
            list(instance.y_down_lz.extract_values().values())
        ).reshape(-1, 24)
        u_down_lz = np.array(
            list(instance.u_down_lz.extract_values().values())
        ).reshape(-1, 24)
        p_down_lz = np.array(
            list(instance.p_down_lz.extract_values().values())
        ).reshape(-1, 24)
        p_up_uz = np.array(list(instance.p_up_uz.extract_values().values())).reshape(
            -1, 24
        )
        u_up_uz = np.array(list(instance.u_up_uz.extract_values().values())).reshape(
            -1, 24
        )
        y_up_uz = np.array(list(instance.y_up_uz.extract_values().values())).reshape(
            -1, 24
        )
        z_up_uz = np.array(list(instance.z_up_uz.extract_values().values())).reshape(
            -1, 24
        )
        z_down_uz = np.array(
            list(instance.z_down_uz.extract_values().values())
        ).reshape(-1, 24)
        y_down_uz = np.array(
            list(instance.y_down_uz.extract_values().values())
        ).reshape(-1, 24)
        u_down_uz = np.array(
            list(instance.u_down_uz.extract_values().values())
        ).reshape(-1, 24)
        p_down_uz = np.array(
            list(instance.p_down_uz.extract_values().values())
        ).reshape(-1, 24)

        # TCL specific
        tzl = np.array(list(instance.tzl.extract_values().values())).reshape(
            -1, 24 * 60
        )
        tzl_base = np.array(list(instance.tzl_base.extract_values().values())).reshape(
            -1
        )
        tzu = np.array(list(instance.tzu.extract_values().values())).reshape(
            -1, 24 * 60
        )
        tzu_base = np.array(list(instance.tzu_base.extract_values().values())).reshape(
            -1
        )
        twl = np.array(list(instance.twl.extract_values().values())).reshape(
            -1, 24 * 60
        )
        twl_base = np.array(list(instance.twl_base.extract_values().values())).reshape(
            -1
        )
        twu = np.array(list(instance.twu.extract_values().values())).reshape(
            -1, 24 * 60
        )
        twu_base = np.array(list(instance.twu_base.extract_values().values())).reshape(
            -1
        )

        twu_data = np.array(list(instance.twu_data.extract_values().values())).reshape(
            -1
        )
        twl_data = np.array(list(instance.twu_data.extract_values().values())).reshape(
            -1
        )

        lambda_spot = np.array(
            list(instance.lambda_spot.extract_values().values())
        ).reshape(-1, 24)

        s = np.array(list(instance.s.extract_values().values())).reshape(-1, 24)
        slack = s.copy()
        up_regulation_event = np.array(
            list(instance.up_regulation_event.extract_values().values())
        ).reshape(-1, 24)
        g_indicator = np.array(list(instance.g.extract_values().values())).reshape(
            -1, 24
        )
        lambda_b = np.array(list(instance.lambda_b.extract_values().values())).reshape(
            -1, 24
        )
        lambda_rp = np.array(
            list(instance.lambda_rp.extract_values().values())
        ).reshape(-1, 24)
        alpha = value(instance.alpha)
        beta = value(instance.beta)

        return InstanceInformation(
            p_base_lz=p_base_lz,
            p_base_uz=p_base_uz,
            pt=pt,
            pt_lz=pt_lz,
            pt_uz=pt_uz,
            p_up_reserve=p_up_reserve,
            p_base=p_base,
            p_up=p_up,
            p_down=p_down,
            s=s,
            slack=slack,
            p_up_lz=p_up_lz,
            u_up_lz=u_up_lz,
            y_up_lz=y_up_lz,
            z_up_lz=z_up_lz,
            z_down_lz=z_down_lz,
            y_down_lz=y_down_lz,
            u_down_lz=u_down_lz,
            p_down_lz=p_down_lz,
            p_up_uz=p_up_uz,
            u_up_uz=u_up_uz,
            y_up_uz=y_up_uz,
            z_up_uz=z_up_uz,
            z_down_uz=z_down_uz,
            y_down_uz=y_down_uz,
            u_down_uz=u_down_uz,
            p_down_uz=p_down_uz,
            tzl=tzl,
            tzl_base=tzl_base,
            tzu=tzu,
            tzu_base=tzu_base,
            twl=twl,
            twl_base=twl_base,
            twu=twu,
            twu_base=twu_base,
            twu_data=twu_data,
            twl_data=twl_data,
            up_regulation_event=up_regulation_event,
            g_indicator=g_indicator,
            lambda_b=lambda_b,
            lambda_rp=lambda_rp,
            lambda_spot=lambda_spot,
            alpha=alpha,
            beta=beta,
        )
    elif case.name == Case.FCR.name:
        # extract solver results
        p_up_reserve = np.array(list(instance.p_up_reserve.extract_values().values()))
        p_up_reserve_lz = np.array(
            list(instance.p_up_reserve_lz.extract_values().values())
        )
        p_up_reserve_uz = np.array(
            list(instance.p_up_reserve_uz.extract_values().values())
        )
        p_base = np.array(list(instance.p_base.extract_values().values()))
        p_base_lz = np.array(list(instance.p_base_lz.extract_values().values()))
        p_base_uz = np.array(list(instance.p_base_uz.extract_values().values()))
        pt = np.array(list(instance.pt.extract_values().values())).reshape(-1, 24 * 60)
        pt_lz = np.array(list(instance.pt_lz.extract_values().values())).reshape(
            -1, 24 * 60
        )
        pt_uz = np.array(list(instance.pt_uz.extract_values().values())).reshape(
            -1, 24 * 60
        )
        p_freq = np.array(list(instance.p_freq.extract_values().values())).reshape(
            -1, 24 * 60
        )
        p_freq_lz = np.array(
            list(instance.p_freq_lz.extract_values().values())
        ).reshape(-1, 24 * 60)
        p_freq_uz = np.array(
            list(instance.p_freq_uz.extract_values().values())
        ).reshape(-1, 24 * 60)

        # TCL specific
        tzl = np.array(list(instance.tzl.extract_values().values())).reshape(
            -1, 24 * 60
        )
        tzl_base = np.array(list(instance.tzl_base.extract_values().values())).reshape(
            -1
        )
        tzu = np.array(list(instance.tzu.extract_values().values())).reshape(
            -1, 24 * 60
        )
        tzu_base = np.array(list(instance.tzu_base.extract_values().values())).reshape(
            -1
        )
        twl = np.array(list(instance.twl.extract_values().values())).reshape(
            -1, 24 * 60
        )
        twl_base = np.array(list(instance.twl_base.extract_values().values())).reshape(
            -1
        )
        twu = np.array(list(instance.twu.extract_values().values())).reshape(
            -1, 24 * 60
        )
        twu_base = np.array(list(instance.twu_base.extract_values().values())).reshape(
            -1
        )
        twu_data = np.array(list(instance.twu_data.extract_values().values())).reshape(
            -1
        )
        twl_data = np.array(list(instance.twu_data.extract_values().values())).reshape(
            -1
        )

        lambda_spot = np.array(
            list(instance.lambda_spot.extract_values().values())
        ).reshape(-1, 24)

        s = np.array(list(instance.s.extract_values().values())).reshape(-1, 24 * 60)
        s_lz = np.array(list(instance.s_lz.extract_values().values())).reshape(
            -1, 24 * 60
        )
        s_uz = np.array(list(instance.s_uz.extract_values().values())).reshape(
            -1, 24 * 60
        )
        s_abs = np.array(list(instance.s.extract_values().values())).reshape(
            -1, 24 * 60
        )
        s_abs_lz = np.array(list(instance.s_lz.extract_values().values())).reshape(
            -1, 24 * 60
        )
        s_abs_uz = np.array(list(instance.s_uz.extract_values().values())).reshape(
            -1, 24 * 60
        )

        lambda_fcr = np.array(
            list(instance.lambda_fcr.extract_values().values())
        ).reshape(-1, 24)
        frequency = np.array(list(instance.freq.extract_values().values())).reshape(
            -1, 24 * 60
        )

        return FCRInstanceInformation(
            p_up_reserve=p_up_reserve,
            p_up_reserve_lz=p_up_reserve_lz,
            p_up_reserve_uz=p_up_reserve_uz,
            p_base=p_base,
            p_base_lz=p_base_lz,
            p_base_uz=p_base_uz,
            pt=pt,
            pt_lz=pt_lz,
            pt_uz=pt_uz,
            p_freq=p_freq,
            p_freq_lz=p_freq_lz,
            p_freq_uz=p_freq_uz,
            s=s,
            s_lz=s_lz,
            s_uz=s_uz,
            s_abs=s_abs,
            s_abs_lz=s_abs_lz,
            s_abs_uz=s_abs_uz,
            frequency=frequency,
            lambda_spot=lambda_spot,
            lambda_fcr=lambda_fcr,
            tzl_base=tzl_base,
            tzl=tzl,
            twl_base=twl_base,
            twl=twl,
            tzu_base=tzu_base,
            tzu=tzu,
            twu_base=twu_base,
            twu=twu,
            twu_data=twu_data,
            twl_data=twl_data,
        )
    else:
        raise NotImplementedError


class Problem:
    def __init__(
        self,
        abstract_model_instance: SolverInstance,
        instance: OptimizationInstance,
    ) -> None:
        self.name = abstract_model_instance.name

        if self.name == Case.mFRR_AND_ENERGY.name:
            data = SolverInstance.mfrr_instance_to_dict(instance)
        elif self.name == Case.FCR.name:
            data = SolverInstance.fcr_instance_to_dict(instance)
        else:
            raise NotImplementedError

        self.model_instance = abstract_model_instance.model.create_instance(data=data)

        self.res_instance: Optional[Any] = None

    def set_objective(self, objective_function: Callable) -> None:
        self.model_instance.objective = Objective(
            rule=objective_function, sense=maximize
        )

    @staticmethod
    def customize_constraints(inst: AbstractModel, one_lambda: bool) -> None:
        if one_lambda:
            inst.lambda_policy_1.deactivate()
            inst.one_lambda_constraint_1.activate()
            inst.one_lambda_constraint_2.activate()
        else:
            inst.lambda_policy_1.activate()
            inst.one_lambda_constraint_1.deactivate()
            inst.one_lambda_constraint_2.deactivate()

    def solve(self, tee: bool = True) -> OptimizationResult:
        self.res_instance, _ = SolverInstance.run_solver(self.model_instance, tee=tee)
        if self.name == Case.mFRR_AND_ENERGY.name:
            opt_result = self.get_mfRR_spot_result(self.res_instance, if_print=tee)
        elif self.name == Case.FCR.name:
            opt_result = self.get_fcr_result(self.res_instance, if_print=tee)
        else:
            raise NotImplementedError
        return opt_result

    @staticmethod
    def get_mfRR_spot_result(
        res_instance: AbstractModel, multiplier: float = 1, if_print: bool = True
    ) -> OptimizationResult:

        p_base = np.array([value(res_instance.p_base[i]) for i in range(24)])
        reserve_payment = (
            value(o_expected_reserve_payment(res_instance)) * multiplier  # type:ignore
        )
        act_payment = (
            value(o_expected_activation_payment(res_instance))  # type:ignore
            * multiplier
        )
        expected_energy_cost = (
            value(o_expected_energy_cost(res_instance)) * multiplier  # type:ignore
        )
        rebound_cost = (
            value(o_expected_rebound_cost(res_instance)) * multiplier  # type:ignore
        )
        expected_power_usage = (
            value(o_expected_energy_consumption(res_instance))  # type:ignore
            * multiplier
        ) / 1000  # mwh
        base_power_usage = (sum(p_base)) * multiplier / 1000  # mwh
        total_cost = -value(o_rule(res_instance)) * multiplier  # type:ignore

        base_cost_today = (
            value(
                sum(  # type:ignore
                    p_base[t]  # type:ignore
                    * (  # type:ignore
                        res_instance.lambda_spot[w, t]  # type:ignore
                        + res_instance.elafgift  # type:ignore
                        + res_instance.tariff[t]  # type:ignore
                    )
                    * (1 + res_instance.moms)  # type:ignore
                    * res_instance.probabilities[w]  # type:ignore
                    for t in res_instance.n_hours  # type:ignore
                    for w in res_instance.nb_scenarios  # type:ignore
                )
            )
            * multiplier
        )
        penalty_cost = value(o_penalty(res_instance)) * multiplier  # type:ignore

        if if_print:
            # print out statistics on earnings/cost
            print(f"Earnings from mFRR reserve: {round(reserve_payment, 1)} DKK")
            print(f"Earnings from mFRR activation: {round(act_payment, 1)} DKK")
            print(f"Earnings from mFRR: {round(reserve_payment + act_payment, 1)} DKK")
            print(f"Base energy usage: {round(base_power_usage, 2)} MWH")
            print(f"Expected energy usage: {round(expected_power_usage, 2)} MWH")
            print(f"Base energy costs today: {round(base_cost_today, 1)} DKK")
            print(f"Expected energy costs: {round(expected_energy_cost, 1)} DKK")
            print(f"Expected total costs: {round(total_cost, 1)} DKK")
            print(f"Expected penalty costs: {round(penalty_cost, 1)} DKK")

            _lambda_b = np.array(
                list(res_instance.lambda_b.extract_values().values())
            ).reshape(-1, 24)
            for i in range(_lambda_b.shape[0]):
                lambda_b = [round(e, 2) if e else 0 for e in _lambda_b[i, :].tolist()]
                print(f"Bid policy: {lambda_b} DKK/kWh")
            p_up_reserve = [
                round(e, 2) for e in res_instance.p_up_reserve.extract_values().values()
            ]
            print(f"p_up reserve policy: {p_up_reserve} kWh")

            print("\n")

        return OptimizationResult(
            reserve_payment=reserve_payment,
            act_payment=act_payment,
            expected_energy_cost=expected_energy_cost,
            rebound_cost=rebound_cost,
            total_cost=total_cost,
            base_cost_today=base_cost_today,
            penalty_cost=penalty_cost,
            battery_capacity=-1,
        )

    @staticmethod
    def get_fcr_result(
        res_instance: AbstractModel, multiplier: float = 1, if_print: bool = True
    ) -> OptimizationResult:
        p_base = np.array([value(res_instance.p_base[i]) for i in range(24)])
        reserve_payment = (
            value(o_expected_fcr_reserve_payment(res_instance)) * multiplier
        )
        act_payment = 0
        # expected_energy_cost = (
        #     value(o_expected_fcr_balance_settlement(res_instance)) * multiplier
        # )
        # expected_energy_cost = 0
        # expected_power_usage = (
        #     value(o_expected_fcr_energy_consumption(res_instance)) * multiplier
        # ) / 1000  # mwh
        # expected_power_usage = 0
        base_power_usage = (sum(p_base)) * multiplier / 1000  # mwh
        total_cost = -value(o_rule_fcr_energy(res_instance)) * multiplier

        base_cost_today = (
            value(
                sum(
                    p_base[t]
                    * (
                        res_instance.lambda_spot[w, t]
                        + res_instance.elafgift
                        + res_instance.tariff[t]
                    )
                    * (1 + res_instance.moms)
                    * res_instance.probabilities[w]
                    for t in res_instance.n_hours
                    for w in res_instance.nb_scenarios
                )
            )
            * multiplier
        )
        total_cost += base_cost_today
        penalty_cost = value(o_fcr_penalty(res_instance)) * multiplier

        if if_print:
            # print out statistics on earnings/cost
            print(f"Earnings from FCR reserve: {round(reserve_payment)} DKK")
            print(f"Base energy usage: {round(base_power_usage, 2)} MWH")
            print(f"Base energy costs today: {round(base_cost_today)} DKK")
            print(f"Expected total costs: {round(total_cost)} DKK")
            print(f"Expected penalty costs: {round(penalty_cost)} DKK")

            p_up_reserve = [
                round(e, 1) for e in res_instance.p_up_reserve.extract_values().values()
            ]
            print(f"p_up reserve policy: {p_up_reserve} kWh")

        return OptimizationResult(
            reserve_payment=reserve_payment,
            act_payment=act_payment,
            expected_energy_cost=0.0,
            rebound_cost=0.0,
            total_cost=total_cost,
            base_cost_today=base_cost_today,
            penalty_cost=penalty_cost,
            battery_capacity=0,
        )
