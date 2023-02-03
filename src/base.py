from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.objective import _o_energy_cost, o_rule, o_rule_fcr_energy

SCENARIO_PATH = "data/scenarios_v2_DK1.csv"


class Case(Enum):
    mFRR_AND_ENERGY = "joint_mfrr_and_energy"
    SPOT = "energy_only"
    FCR = "fcr"


OBJECTIVE_FUNCTION = {
    Case.mFRR_AND_ENERGY.name: o_rule,
    Case.SPOT.name: _o_energy_cost,
    Case.FCR.name: o_rule_fcr_energy,
}


# NOTE: NOT USED!!!
@dataclass
class OptimizationInstanceZincFurnace:
    lambda_mfrr: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    up_regulation_event: np.ndarray
    probabilities: np.ndarray

    elafgift: float
    moms: float
    tariff: np.ndarray

    # TCL specific data
    regime: np.ndarray

    p_base_lz: np.ndarray
    setpoint_lz: float
    tzl_data: np.ndarray

    p_base_uz: np.ndarray
    setpoint_uz: float
    tzu_data: np.ndarray

    p_base: np.ndarray

    ta: float
    Czu: float
    Czl: float
    Cwu: float
    Cwl: float
    Rww: float
    Rwz1: float
    Rwz2: float
    Rzuzl: float
    Rwua1: float
    Rwua2: float
    Rwla1: float
    Rwla2: float

    p_nom: float
    p_min: float
    delta_max: float

    one_lambda: bool

    dt: float
    n_steps: int
    n_hours: int = field(init=False)
    hour_steps: int = field(init=False)
    nb_scenarios: int = field(init=False)

    max_up_time: int
    min_up_time: int

    lambda_fcr: Optional[np.ndarray] = None
    frequency: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.nb_scenarios = self.up_regulation_event.shape[0]
        assert self.n_steps * self.dt % 1 == 0
        self.n_hours = int(self.n_steps * self.dt)
        assert 1 / self.dt % 1 == 0
        self.hour_steps = int(1 / self.dt)
        assert self.tariff.shape[0] == self.n_hours

    def reduce_instance(self, nb: int = 1) -> None:
        # Reduce instance to "nb" scenarios.
        # Function can, e.g., used for oos evaluation
        assert nb >= 1, "nb must be >= 1"
        self.nb_scenarios = nb
        self.lambda_mfrr = self.lambda_mfrr[0:nb, :]
        self.lambda_rp = self.lambda_rp[0:nb, :]
        self.lambda_spot = self.lambda_spot[0:nb, :]
        self.up_regulation_event = self.up_regulation_event[0:nb, :]
        self.probabilities = np.array([1 / nb for _ in range(nb)])
        if self.lambda_fcr is not None:
            self.lambda_fcr = self.lambda_fcr[0:nb, :]
        if self.frequency is not None:
            self.frequency = self.frequency[0:nb, :]

    def convert_to_robust_scenario(self) -> None:
        self.reduce_instance()
        self.lambda_rp = self.lambda_spot * 2
        self.up_regulation_event = np.ones(shape=(1, 24))

    def convert_to_spot_price_case(self) -> None:
        self.reduce_instance()
        self.old_lambda_rp = self.lambda_rp
        self.lambda_rp = self.lambda_spot + 10
        self.up_regulation_event = np.ones(shape=self.lambda_rp.shape)
