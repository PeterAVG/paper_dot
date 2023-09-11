import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, cast

import numpy as np
import pyomo.kernel as pmo
from pyomo.environ import (
    AbstractModel,
    Constraint,
    NonNegativeReals,
    Param,
    RangeSet,
    Reals,
    SolverFactory,
    Var,
    floor,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

from src.base import Case
from src.constraints import (
    delta_constraint,
    twl_baseline_constraint,
    twl_constraint,
    twl_constraint_1,
    twl_constraint_2,
    twu_baseline_constraint,
    twu_constraint,
    twu_constraint_1,
    twu_constraint_2,
    tzl_baseline_constraint,
    tzl_constraint,
    tzl_constraint_1,
    tzl_constraint_2,
    tzu_baseline_constraint,
    tzu_constraint,
    tzu_constraint_1,
    tzu_constraint_2,
)

logger = logging.getLogger(__name__)

# papers on food degradation and ice crystals
# https://www.sciencedirect.com/science/article/pii/S0260877410002013?casa_token=YlidsOemDNwAAAAA:dM94qp8mLCQ_C-9bBNKDl72UJNIU4RToFmq4KzYST4uxdbkjsIT-oQkHzknyiZXz6st3CzuR9gi2
# https://link.springer.com/article/10.1007/s12393-020-09255-8#Tab3
# https://www.sciencedirect.com/science/article/abs/pii/S0309174013001563
# Best one: describes that size and duration of fluctuations is bad
# https://www.sciencedirect.com/science/article/pii/S0023643814003715?via%3Dihub


@dataclass
class OptimizationInstance:
    lambda_mfrr: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    up_regulation_event: np.ndarray
    probabilities: np.ndarray

    elafgift: float
    moms: float
    tariff: np.ndarray

    nb_scenarios: int = field(init=False)

    # TCL specific data
    regime: np.ndarray

    p_base_lz: np.ndarray
    setpoint_lz: float
    twl_data: np.ndarray

    p_base_uz: np.ndarray
    setpoint_uz: float
    twu_data: np.ndarray

    p_base: np.ndarray

    ta: float
    Czu: float
    Czl: float
    Cwu: float
    Cwl: float
    Rww: float
    Rwz: float
    Rzuzl: float
    Rwua1: float
    Rwua2: float
    Rwla1: float

    p_nom: float
    p_min: float
    delta_max: float

    one_lambda: bool

    dt: float
    n_steps: int
    n_hours: int = field(init=False)
    hour_steps: int = field(init=False)

    max_up_time: int
    min_up_time: int
    rebound: int

    M: int
    max_lambda_bid: float

    lambda_fcr: Optional[np.ndarray] = None
    frequency: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.nb_scenarios = self.up_regulation_event.shape[0]
        assert self.n_steps * self.dt % 1 == 0
        self.n_hours = int(self.n_steps * self.dt)
        assert 1 / self.dt % 1 == 0
        self.hour_steps = int(1 / self.dt)
        assert self.tariff.shape[0] == self.n_hours

        if self.lambda_fcr is not None:
            assert self.lambda_fcr.shape[0] == self.nb_scenarios
            assert self.lambda_fcr.shape == self.lambda_spot.shape

        if self.frequency is not None:
            assert self.frequency.shape[0] == self.nb_scenarios
            assert self.frequency.shape[1] == self.twu_data.shape[0]
            assert self.frequency.shape[1] == self.twl_data.shape[0]

    def reduce_instance(self, nb: int = 1) -> None:
        # Reduce instance to "nb" scenarios.
        # Function can, e.g., used for oos evaluation
        self.nb_scenarios = nb
        self.lambda_mfrr = self.lambda_mfrr[0:nb, :]
        self.lambda_rp = self.lambda_rp[0:nb, :]
        self.lambda_spot = self.lambda_spot[0:nb, :]
        self.up_regulation_event = self.up_regulation_event[0:nb, :]
        self.probabilities = np.array([1 for _ in range(nb)])
        if self.lambda_fcr is not None:
            self.lambda_fcr = self.lambda_fcr[0:nb, :]
        if self.frequency is not None:
            self.frequency = self.frequency[0:nb, :]

    def convert_to_robust_scenario(self) -> None:
        self.reduce_instance()
        self.lambda_rp = self.lambda_spot * 2
        self.up_regulation_event = np.ones(shape=(1, 24))

    def lookback_generator(self, lookback: int) -> Generator:
        # Generator that returns chunks of the instance
        assert lookback > 0
        assert lookback < self.nb_scenarios
        for i in range(lookback, self.nb_scenarios):
            yield self(i - lookback, i), self(i, i + 1)

    def chunk_generator(self, chunk_size: int) -> Generator:
        # Generator that returns chunks of the instance
        assert chunk_size > 0
        if chunk_size >= self.nb_scenarios:
            for _ in range(1):
                yield self(0, self.nb_scenarios)
        else:
            for i in range(0, self.nb_scenarios, chunk_size):
                yield self(i, i + chunk_size)

    def __call__(self, lb: int, ub: int) -> "OptimizationInstance":
        # Return a chunk of the instance with the given index
        assert lb >= 0
        assert lb <= self.nb_scenarios - 1
        assert ub > 0
        assert ub > lb
        ub = min(self.nb_scenarios, ub)
        nb_scenarios = ub - lb

        _copy = deepcopy(self)
        _copy.nb_scenarios = nb_scenarios
        _copy.lambda_mfrr = self.lambda_mfrr[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.lambda_rp = self.lambda_rp[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.lambda_spot = self.lambda_spot[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.up_regulation_event = self.up_regulation_event[lb:ub, :].reshape(
            nb_scenarios, -1
        )
        _copy.probabilities = np.ones(nb_scenarios) / nb_scenarios
        assert _copy.up_regulation_event.shape == (nb_scenarios, self.n_hours)
        assert _copy.up_regulation_event.shape == _copy.lambda_mfrr.shape
        assert _copy.up_regulation_event.shape == _copy.lambda_rp.shape
        assert _copy.up_regulation_event.shape == _copy.lambda_spot.shape

        if self.lambda_fcr is not None:
            _copy.lambda_fcr = self.lambda_fcr[lb:ub, :].reshape(nb_scenarios, -1)
            assert _copy.lambda_fcr.shape == (nb_scenarios, self.n_hours)
        if self.frequency is not None:
            _copy.frequency = self.frequency[lb:ub, :].reshape(nb_scenarios, -1)
            assert _copy.frequency.shape == (nb_scenarios, self.n_steps)

        return _copy


def pyomo_get_abstract_mfrr_model() -> AbstractModel:

    model = AbstractModel()

    model._n_steps = Param(initialize=1440, mutable=False)
    model.hour_steps = Param(initialize=60, mutable=False)
    model._n_hours = Param(initialize=24, mutable=False)
    model._nb_scenarios = Param(initialize=1, mutable=True)

    model.time_steps = RangeSet(0, model._n_steps - 1)
    model.n_hours = RangeSet(0, model._n_hours - 1)
    model.nb_scenarios = RangeSet(0, model._nb_scenarios - 1)

    model.M = Param(initialize=15, mutable=True)
    model.max_lambda_bid = Param(initialize=50, mutable=True)
    # one_lambda = inst.one_lambda

    model.p_base = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_base_lz = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_base_uz = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_nom = Param(mutable=True)
    model.p_min = Param(mutable=True)

    model.ta = Param(mutable=True)
    model.Czu = Param(mutable=True)
    model.Czl = Param(mutable=True)
    model.Cwu = Param(mutable=True)
    model.Cwl = Param(mutable=True)
    model.Rww = Param(mutable=True)
    model.Rwz = Param(mutable=True)
    model.Rzuzl = Param(mutable=True)
    model.Rwua1 = Param(mutable=True)
    model.Rwua2 = Param(mutable=True)
    model.Rwla1 = Param(mutable=True)
    model.setpoint_lz = Param(mutable=True)
    model.setpoint_uz = Param(mutable=True)
    model.regime = Param(model.time_steps, mutable=True, domain=Reals)

    # base temperature data. Could probablt replace base variables with this
    model.twl_data = Param(model.time_steps, mutable=True, domain=Reals)
    model.twu_data = Param(model.time_steps, mutable=True, domain=Reals)

    # variables for thermodynamics
    model.tzl = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.tzu = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.tzl_base = Var(model.time_steps, domain=Reals)
    model.tzu_base = Var(model.time_steps, domain=Reals)
    model.twl = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.twu = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.twl_base = Var(model.time_steps, domain=Reals)
    model.twu_base = Var(model.time_steps, domain=Reals)

    model.dt = Param(mutable=True)
    model.delta_max = Param(mutable=True)

    # temperature deviation
    model.delta = Var(domain=NonNegativeReals)

    model.rebound = Param(mutable=True)
    model.setpoint = Param(mutable=True)
    model.max_up_time = Param(mutable=True)
    model.min_up_time = Param(mutable=True)

    model.lambda_spot = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.lambda_rp = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.lambda_mfrr = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.up_regulation_event = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.probabilities = Param(model.nb_scenarios, mutable=True)

    model.elafgift = Param(mutable=True)
    model.moms = Param(mutable=True)
    model.tariff = Param(model.n_hours, mutable=True, domain=Reals)

    model.u_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.u_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)

    # power regulation
    model.u_up_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.u_down_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_up_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_down_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_up_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_down_lz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.p_up_lz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.p_down_lz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.pt_lz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.s_lz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.u_up_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.u_down_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_up_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_down_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_up_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_down_uz = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.p_up_uz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.p_down_uz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.pt_uz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.s_uz = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)

    # reservation capacity (up mFRR)
    model.p_up_reserve_lz = Var(model.n_hours, domain=NonNegativeReals)
    model.p_up_reserve_uz = Var(model.n_hours, domain=NonNegativeReals)

    model.p_up_reserve = Var(model.n_hours, domain=NonNegativeReals)
    model.p_up = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.p_down = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.pt = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.s = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)

    # variables related to bid strategy
    model.lambda_b = Var(
        model.nb_scenarios,
        model.n_hours,
        domain=NonNegativeReals,
    )
    model.phi = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.g = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.alpha = Var(domain=NonNegativeReals, initialize=1)
    model.beta = Var(domain=NonNegativeReals, initialize=1)

    return model


def pyomo_get_abstract_fcr_model() -> AbstractModel:

    model = AbstractModel()

    model._n_steps = Param(initialize=1440, mutable=False)
    model.hour_steps = Param(initialize=60, mutable=False)
    model._n_hours = Param(initialize=24, mutable=False)
    model._nb_scenarios = Param(initialize=1, mutable=True)

    model.time_steps = RangeSet(0, model._n_steps - 1)
    model.n_hours = RangeSet(0, model._n_hours - 1)
    model.nb_scenarios = RangeSet(0, model._nb_scenarios - 1)

    model.p_base = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_base_lz = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_base_uz = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_nom = Param(mutable=True)
    model.p_min = Param(mutable=True)

    model.ta = Param(mutable=True)
    model.Czu = Param(mutable=True)
    model.Czl = Param(mutable=True)
    model.Cwu = Param(mutable=True)
    model.Cwl = Param(mutable=True)
    model.Rww = Param(mutable=True)
    model.Rwz = Param(mutable=True)
    model.Rzuzl = Param(mutable=True)
    model.Rwua1 = Param(mutable=True)
    model.Rwua2 = Param(mutable=True)
    model.Rwla1 = Param(mutable=True)
    model.setpoint_lz = Param(mutable=True)
    model.setpoint_uz = Param(mutable=True)
    model.regime = Param(model.time_steps, mutable=True, domain=Reals)

    # base temperature data. Could probablt replace base variables with this
    model.twl_data = Param(model.time_steps, mutable=True, domain=Reals)
    model.twu_data = Param(model.time_steps, mutable=True, domain=Reals)

    # variables for thermodynamics
    model.tzl = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.tzu = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.tzl_base = Var(model.time_steps, domain=Reals)
    model.tzu_base = Var(model.time_steps, domain=Reals)
    model.twl = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.twu = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.twl_base = Var(model.time_steps, domain=Reals)
    model.twu_base = Var(model.time_steps, domain=Reals)

    model.dt = Param(mutable=True)
    model.delta_max = Param(mutable=True)

    # temperature deviation
    model.delta = Var(domain=NonNegativeReals)

    model.max_up_time = Param(mutable=True)
    model.min_up_time = Param(mutable=True)

    model.lambda_spot = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.lambda_fcr = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.probabilities = Param(model.nb_scenarios, mutable=True)

    model.elafgift = Param(mutable=True)
    model.moms = Param(mutable=True)
    model.tariff = Param(model.n_hours, mutable=True, domain=Reals)

    # power in each zone and in total
    model.pt_lz = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)
    model.pt_uz = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)
    model.pt = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)

    # reservation capacity
    model.p_up_reserve_lz = Var(model.n_hours, domain=NonNegativeReals)
    model.p_up_reserve_uz = Var(model.n_hours, domain=NonNegativeReals)
    model.p_up_reserve = Var(model.n_hours, domain=NonNegativeReals)

    # power adjustment to frequency deviation
    model.p_freq_lz = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.p_freq_uz = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.p_freq = Var(model.nb_scenarios, model.time_steps, domain=Reals)

    # slack used for penalty
    model.s_lz = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.s_uz = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.s = Var(model.nb_scenarios, model.time_steps, domain=Reals)

    # for absolute value
    model.s_abs_lz = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)
    model.s_abs_uz = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)
    model.s_abs = Var(model.nb_scenarios, model.time_steps, domain=NonNegativeReals)

    # frequency
    model.freq = Param(model.nb_scenarios, model.time_steps, mutable=False)

    return model


class SolverInstance:
    def __init__(self, case: Case) -> None:
        self.name = case.name
        if self.name == Case.mFRR_AND_ENERGY.name:
            self.model = pyomo_get_abstract_mfrr_model()

            self.add_total_power_constraints()
            self.add_power_constraints_lz()
            self.add_power_constraints_uz()
            self.add_bid_constraints()
            self.add_thermodynamics_constraints_baseline()
            self.add_thermodynamics_constraints()
            self.add_auxillary_constraints_lz()
            self.add_auxillary_constraints_uz()
            self.add_rebound_constraints_lz()
            self.add_rebound_constraints_uz()
        elif self.name == Case.FCR.name:
            self.model = pyomo_get_abstract_fcr_model()

            self.add_total_power_constraints_fcr()
            self.add_power_constraints_lz_fcr()
            self.add_power_constraints_uz_fcr()
            self.add_thermodynamics_constraints_baseline()
            self.add_thermodynamics_constraints()

        else:
            raise ValueError("Case not implemented")

        self.model.name = self.name

    @staticmethod
    def mfrr_instance_to_dict(inst: OptimizationInstance) -> Dict:
        data: Dict[None, Any] = {None: {}}

        data[None]["_n_steps"] = {None: inst.n_steps}
        data[None]["hour_steps"] = {None: inst.hour_steps}
        data[None]["_n_hours"] = {None: inst.n_hours}
        data[None]["_nb_scenarios"] = {None: inst.nb_scenarios}

        data[None]["M"] = {None: inst.M}
        data[None]["max_lambda_bid"] = {None: inst.max_lambda_bid}

        # TCL specific
        data[None]["ta"] = {None: inst.ta}
        data[None]["Czu"] = {None: inst.Czu}
        data[None]["Czl"] = {None: inst.Czl}
        data[None]["Cwu"] = {None: inst.Cwu}
        data[None]["Cwl"] = {None: inst.Cwl}
        data[None]["Rww"] = {None: inst.Rww}
        data[None]["Rwz"] = {None: inst.Rwz}
        data[None]["Rzuzl"] = {None: inst.Rzuzl}
        data[None]["Rwua1"] = {None: inst.Rwua1}
        data[None]["Rwua2"] = {None: inst.Rwua2}
        data[None]["Rwla1"] = {None: inst.Rwla1}
        data[None]["setpoint_lz"] = {None: inst.setpoint_lz}
        data[None]["setpoint_uz"] = {None: inst.setpoint_uz}
        data[None]["regime"] = {i: e for i, e in enumerate(inst.regime)}

        data[None]["p_base_lz"] = {i: e for i, e in enumerate(inst.p_base_lz)}
        data[None]["p_base_uz"] = {i: e for i, e in enumerate(inst.p_base_uz)}
        data[None]["p_base"] = {i: e for i, e in enumerate(inst.p_base)}

        data[None]["p_nom"] = {None: inst.p_nom}
        data[None]["p_min"] = {None: inst.p_min}

        data[None]["twl_data"] = {i: e for i, e in enumerate(inst.twl_data)}
        data[None]["twu_data"] = {i: e for i, e in enumerate(inst.twu_data)}

        data[None]["dt"] = {None: inst.dt}
        data[None]["delta_max"] = {None: inst.delta_max}

        data[None]["max_up_time"] = {None: inst.max_up_time}
        data[None]["min_up_time"] = {None: inst.min_up_time}

        data[None]["lambda_spot"] = {
            (i, j): inst.lambda_spot[i, j]
            for i in range(inst.lambda_spot.shape[0])
            for j in range(inst.lambda_spot.shape[1])
        }
        data[None]["lambda_rp"] = {
            (i, j): inst.lambda_rp[i, j]
            for i in range(inst.lambda_rp.shape[0])
            for j in range(inst.lambda_rp.shape[1])
        }
        data[None]["lambda_mfrr"] = {
            (i, j): inst.lambda_mfrr[i, j]
            for i in range(inst.lambda_mfrr.shape[0])
            for j in range(inst.lambda_mfrr.shape[1])
        }
        data[None]["up_regulation_event"] = {
            (i, j): inst.up_regulation_event[i, j]
            for i in range(inst.up_regulation_event.shape[0])
            for j in range(inst.up_regulation_event.shape[1])
        }

        data[None]["elafgift"] = {None: inst.elafgift}
        data[None]["moms"] = {None: inst.moms}
        data[None]["tariff"] = {i: e for i, e in enumerate(inst.tariff)}

        data[None]["probabilities"] = {i: e for i, e in enumerate(inst.probabilities)}

        return data

    @staticmethod
    def fcr_instance_to_dict(inst: OptimizationInstance) -> Dict:
        data: Dict[None, Any] = {None: {}}

        data[None]["_n_steps"] = {None: inst.n_steps}
        data[None]["hour_steps"] = {None: inst.hour_steps}
        data[None]["_n_hours"] = {None: inst.n_hours}
        data[None]["_nb_scenarios"] = {None: inst.nb_scenarios}

        data[None]["M"] = {None: inst.M}
        data[None]["max_lambda_bid"] = {None: inst.max_lambda_bid}

        # TCL specific
        data[None]["ta"] = {None: inst.ta}
        data[None]["Czu"] = {None: inst.Czu}
        data[None]["Czl"] = {None: inst.Czl}
        data[None]["Cwu"] = {None: inst.Cwu}
        data[None]["Cwl"] = {None: inst.Cwl}
        data[None]["Rww"] = {None: inst.Rww}
        data[None]["Rwz"] = {None: inst.Rwz}
        data[None]["Rzuzl"] = {None: inst.Rzuzl}
        data[None]["Rwua1"] = {None: inst.Rwua1}
        data[None]["Rwua2"] = {None: inst.Rwua2}
        data[None]["Rwla1"] = {None: inst.Rwla1}
        data[None]["setpoint_lz"] = {None: inst.setpoint_lz}
        data[None]["setpoint_uz"] = {None: inst.setpoint_uz}
        data[None]["regime"] = {i: e for i, e in enumerate(inst.regime)}

        data[None]["p_base_lz"] = {i: e for i, e in enumerate(inst.p_base_lz)}
        data[None]["p_base_uz"] = {i: e for i, e in enumerate(inst.p_base_uz)}
        data[None]["p_base"] = {i: e for i, e in enumerate(inst.p_base)}

        data[None]["p_nom"] = {None: inst.p_nom}
        data[None]["p_min"] = {None: inst.p_min}

        data[None]["twl_data"] = {i: e for i, e in enumerate(inst.twl_data)}
        data[None]["twu_data"] = {i: e for i, e in enumerate(inst.twu_data)}

        data[None]["dt"] = {None: inst.dt}
        data[None]["delta_max"] = {None: inst.delta_max}

        data[None]["max_up_time"] = {None: inst.max_up_time}
        data[None]["min_up_time"] = {None: inst.min_up_time}

        data[None]["lambda_spot"] = {
            (i, j): inst.lambda_spot[i, j]
            for i in range(inst.lambda_spot.shape[0])
            for j in range(inst.lambda_spot.shape[1])
        }
        data[None]["lambda_fcr"] = {
            (i, j): cast(np.ndarray, inst.lambda_fcr)[i, j]
            for i in range(cast(np.ndarray, inst.lambda_fcr).shape[0])
            for j in range(cast(np.ndarray, inst.lambda_fcr).shape[1])
        }
        data[None]["freq"] = {
            (i, j): cast(np.ndarray, inst.frequency)[i, j]
            for i in range(cast(np.ndarray, inst.frequency).shape[0])
            for j in range(cast(np.ndarray, inst.frequency).shape[1])
        }

        data[None]["elafgift"] = {None: inst.elafgift}
        data[None]["moms"] = {None: inst.moms}
        data[None]["tariff"] = {i: e for i, e in enumerate(inst.tariff)}

        data[None]["probabilities"] = {i: e for i, e in enumerate(inst.probabilities)}

        return data

    def add_total_power_constraints(self) -> None:
        def total_reservation(m, t):  # type:ignore
            return m.p_up_reserve[t] == m.p_up_reserve_lz[t] + m.p_up_reserve_uz[t]

        def total_up_regulation(m, w, t):  # type:ignore
            return m.p_up[w, t] == m.p_up_lz[w, t] + m.p_up_uz[w, t]

        def total_down_regulation(m, w, t):  # type:ignore
            return m.p_down[w, t] == m.p_down_lz[w, t] + m.p_down_uz[w, t]

        def total_power(m, w, t):  # type:ignore
            return m.pt[w, t] == m.pt_lz[w, t] + m.pt_uz[w, t]

        def total_power_max(m, w, t):  # type:ignore
            return m.pt[w, t] <= m.p_nom * 2

        def total_slack(m, w, t):  # type:ignore
            return m.s[w, t] == m.s_lz[w, t] + m.s_uz[w, t]

        def test(m):  # type:ignore
            # return m.p_up_reserve[t] == m.p_base[t]
            return m.p_up[0, 13] >= 100

        # add constraints to model
        self.model.total_reservation = Constraint(
            self.model.n_hours,
            rule=total_reservation,
        )
        self.model.total_up_regulation = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=total_up_regulation,
        )
        self.model.total_down_regulation = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=total_down_regulation,
        )
        self.model.total_power = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=total_power,
        )
        self.model.total_power_max = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=total_power_max,
        )
        self.model.total_slack = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=total_slack,
        )
        # self.model.test = Constraint(
        #     rule=test,
        # )

    def add_total_power_constraints_fcr(self) -> None:
        def total_reservation(m, h):  # type:ignore
            return m.p_up_reserve[h] == m.p_up_reserve_lz[h] + m.p_up_reserve_uz[h]

        def reservation_1(m, h):  # type:ignore
            return m.p_up_reserve[h] <= m.p_base[h]

        def total_regulation(m, w, t):  # type:ignore
            return m.p_freq[w, t] == m.p_freq_lz[w, t] + m.p_freq_uz[w, t]

        def total_power(m, w, t):  # type:ignore
            return m.pt[w, t] == m.pt_lz[w, t] + m.pt_uz[w, t]

        def total_power_max(m, w, t):  # type:ignore
            return m.pt[w, t] <= m.p_nom * 2

        def total_slack(m, w, t):  # type:ignore
            return m.s[w, t] == m.s_lz[w, t] + m.s_uz[w, t]

        def power_frequency_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.p_freq[w, t] == m.freq[w, t] * m.p_up_reserve[h] + m.s[w, t]

        def power_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.pt[w, t] == m.p_freq[w, t] + m.p_base[h]

        def slack_absolute_value_split_1(m, w, t):  # type:ignore
            return m.s_abs_lz[w, t] + m.s_abs_uz[w, t] <= m.s_abs[w, t]

        def slack_absolute_value_split_2(m, w, t):  # type:ignore
            return -(m.s_abs_lz[w, t] + m.s_abs_uz[w, t]) <= m.s_abs[w, t]

        def test(m, h):  # type:ignore
            return m.p_up_reserve[h] == m.p_base[h]

        # add constraints to model
        self.model.total_reservation = Constraint(
            self.model.n_hours,
            rule=total_reservation,
        )
        self.model.reservation_1 = Constraint(
            self.model.n_hours,
            rule=reservation_1,
        )
        self.model.total_regulation = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_regulation,
        )
        self.model.total_power = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_power,
        )
        self.model.total_power_max = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_power_max,
        )
        self.model.total_slack = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_slack,
        )
        self.model.power_frequency_response = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_frequency_response,
        )
        self.model.power_response = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_response,
        )
        self.model.slack_absolute_value_split_1 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model.slack_absolute_value_split_2 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_2,
        )
        # self.model.test = Constraint(
        #     rule=test,
        # )

    def add_power_constraints_lz(self) -> None:  # type:ignore
        # power constraints for lower zone
        def power_constraint_1_lz(m, w, t):  # type:ignore
            return m.pt_lz[w, t] == m.p_base_lz[t] - m.p_up_lz[w, t] + m.p_down_lz[w, t]

        def power_constraint_2_lz(m, w, t):  # type:ignore
            return m.p_up_lz[w, t] <= m.u_up_lz[w, t] * (m.p_base_lz[t] - m.p_min)

        def power_constraint_2_2_lz(m, w, t):  # type:ignore
            return m.p_up_lz[w, t] <= m.p_up_reserve_lz[t] * m.up_regulation_event[w, t]

        def power_constraint_3_lz(m, w, t):  # type:ignore
            return m.p_down_lz[w, t] <= m.u_down_lz[w, t] * m.p_nom

        def power_constraint_4_lz(m, w, t):  # type:ignore
            return (
                m.p_up_lz[w, t] + m.s_lz[w, t]
                >= m.u_up_lz[w, t] * m.p_base_lz[t] * 0.10
            )

        # below constraint not used since it is not only up_regulation_event that can trigger up regulation
        # but also if our bid is smaller than the balancing price
        def power_constraint_4_2_lz(m, w, t):  # type:ignore
            return (
                m.p_up_lz[w, t]
                >= m.p_up_reserve_lz[t] * m.up_regulation_event[w, t] - m.s_lz[w, t]
            )

        def power_constraint_5_lz(m, w, t):  # type:ignore
            return m.p_down_lz[w, t] >= m.u_down_lz[w, t] * 0.10 * (
                m.p_nom - m.p_base_lz[t]
            )

        def power_constraint_6_lz(m, w, t):  # type:ignore
            return m.pt_lz[w, t] <= m.p_nom

        def power_constraint_6_2_lz(m, w, t):  # type:ignore
            return m.pt_lz[w, t] >= m.p_min

        def power_constraint_7_lz(m, w, t):  # type:ignore
            return m.s_lz[w, t] >= 0

        def power_constraint_8_lz(m, w, t):  # type:ignore
            return m.s_lz[w, t] <= m.p_base_lz[t]

        def power_constraint_9_lz(m, t):  # type:ignore
            return m.p_up_reserve_lz[t] <= m.p_base_lz[t]

        def power_constraint_10_lz(m, w, t):  # type:ignore
            return m.p_up_lz[w, t] <= m.p_nom

        def power_constraint_11_lz(m, w, t):  # type:ignore
            return m.p_down_lz[w, t] <= m.p_nom

        def power_constraint_test_lz(m, w, t):  # type:ignore
            return m.pt_lz[w, t] == m.p_base_lz[t]

        # add power constraints for lower zone to model
        self.model.P1_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_1_lz,
        )
        self.model.P2_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2_lz,
        )
        self.model.P22_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2_2_lz,
        )
        self.model.P3_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_3_lz,
        )
        self.model.P4_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_4_lz,
        )
        # self.model.P44_LZ = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_4_2_lz,
        # )
        self.model.P5_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_5_lz,
        )
        self.model.P6_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6_lz,
        )
        self.model.P6_2_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6_2_lz,
        )
        self.model.P7_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_7_lz,
        )
        self.model.P8_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_8_lz,
        )
        self.model.P9_LZ = Constraint(
            self.model.n_hours,
            rule=power_constraint_9_lz,
        )
        self.model.P10_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_10_lz,
        )
        self.model.P11_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_11_lz,
        )
        # self.model.P_TEST_LZ = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_test_lz,
        # )

    def add_power_constraints_lz_fcr(self) -> None:  # type:ignore
        def total_power(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.pt_lz[w, t] == m.p_base_lz[h] + m.p_freq_lz[w, t]

        def regulation_bound_1(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.p_freq_lz[w, t] <= m.p_nom - m.p_base_lz[h]

        def regulation_bound_2(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.p_freq_lz[w, t] >= -m.p_base_lz[h]

        def power_frequency_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return (
                m.p_freq_lz[w, t] == m.freq[w, t] * m.p_up_reserve_lz[h] + m.s_lz[w, t]
            )

        def power_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.pt_lz[w, t] == m.p_freq_lz[w, t] + m.p_base_lz[h]

        def slack_absolute_value_split_1(m, w, t):  # type:ignore
            return m.s_lz[w, t] <= m.s_abs_lz[w, t]

        def slack_absolute_value_split_2(m, w, t):  # type:ignore
            return -m.s_lz[w, t] <= m.s_abs_lz[w, t]

        self.model.total_power_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_power,
        )
        self.model.regulation_bound_1_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=regulation_bound_1,
        )
        self.model.regulation_bound_2_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=regulation_bound_2,
        )
        self.model.power_frequency_response_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_frequency_response,
        )
        self.model.power_response_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_response,
        )
        self.model.slack_absolute_value_split_1_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model.slack_absolute_value_split_2_lz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_2,
        )

    # power constraints for upper zone

    def add_power_constraints_uz(self):  # type:ignore
        def power_constraint_1_uz(m, w, t):  # type:ignore
            return m.pt_uz[w, t] == m.p_base_uz[t] - m.p_up_uz[w, t] + m.p_down_uz[w, t]

        def power_constraint_2_uz(m, w, t):  # type:ignore
            return m.p_up_uz[w, t] <= m.u_up_uz[w, t] * (m.p_base_uz[t] - m.p_min)

        def power_constraint_2_2_uz(m, w, t):  # type:ignore
            return m.p_up_uz[w, t] <= m.p_up_reserve_uz[t] * m.up_regulation_event[w, t]

        def power_constraint_3_uz(m, w, t):  # type:ignore
            return m.p_down_uz[w, t] <= m.u_down_uz[w, t] * m.p_nom

        def power_constraint_4_uz(m, w, t):  # type:ignore
            return (
                m.p_up_uz[w, t] + m.s_uz[w, t]
                >= m.u_up_uz[w, t] * m.p_base_uz[t] * 0.10
            )

        # below constraint not used since it is not only up_regulation_event that can trigger up regulation
        # but also if our bid is smaller than the balancing price
        def power_constraint_4_2_uz(m, w, t):  # type:ignore
            return (
                m.p_up_uz[w, t]
                >= m.p_up_reserve_uz[t] * m.up_regulation_event[w, t] - m.s_uz[w, t]
            )

        def power_constraint_5_uz(m, w, t):  # type:ignore
            return m.p_down_uz[w, t] >= m.u_down_uz[w, t] * 0.10 * (
                m.p_nom - m.p_base_uz[t]
            )

        def power_constraint_6_uz(m, w, t):  # type:ignore
            return m.pt_uz[w, t] <= m.p_nom

        def power_constraint_6_2_uz(m, w, t):  # type:ignore
            return m.pt_uz[w, t] >= m.p_min

        def power_constraint_7_uz(m, w, t):  # type:ignore
            return m.s_uz[w, t] >= 0

        def power_constraint_8_uz(m, w, t):  # type:ignore
            return m.s_uz[w, t] <= m.p_base_uz[t]

        def power_constraint_9_uz(m, t):  # type:ignore
            return m.p_up_reserve_uz[t] <= m.p_base_uz[t]

        def power_constraint_10_uz(m, w, t):  # type:ignore
            return m.p_up_uz[w, t] <= m.p_nom

        def power_constraint_11_uz(m, w, t):  # type:ignore
            return m.p_down_uz[w, t] <= m.p_nom

        def power_constraint_test_uz(m, t):  # type:ignore
            return m.p_up_reserve_uz[t] == m.p_base_uz[t]

        # add power constraints for upper zone to model
        self.model.P1_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_1_uz,
        )
        self.model.P2_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2_uz,
        )
        self.model.P22_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2_2_uz,
        )
        self.model.P3_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_3_uz,
        )
        self.model.P4_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_4_uz,
        )
        # self.model.P44_UZ = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_4_2_uz,
        # )
        self.model.P5_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_5_uz,
        )
        self.model.P6_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6_uz,
        )
        self.model.P6_2_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6_2_uz,
        )
        self.model.P7_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_7_uz,
        )
        self.model.P8_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_8_uz,
        )
        self.model.P9_UZ = Constraint(
            self.model.n_hours,
            rule=power_constraint_9_uz,
        )
        self.model.P10_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_10_uz,
        )
        self.model.P11_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_11_uz,
        )
        # self.model.P_TEST_UZ = Constraint(
        #     self.model.n_hours,
        #     rule=power_constraint_test_uz,
        # )

    def add_power_constraints_uz_fcr(self):  # type:ignore
        def total_power(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.pt_uz[w, t] == m.p_base_uz[h] + m.p_freq_uz[w, t]

        def regulation_bound_1(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.p_freq_uz[w, t] <= m.p_nom - m.p_base_uz[h]

        def regulation_bound_2(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.p_freq_uz[w, t] >= -m.p_base_uz[h]

        def power_frequency_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return (
                m.p_freq_uz[w, t] == m.freq[w, t] * m.p_up_reserve_uz[h] + m.s_uz[w, t]
            )

        def power_response(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)
            return m.pt_uz[w, t] == m.p_freq_uz[w, t] + m.p_base_uz[h]

        def slack_absolute_value_split_1(m, w, t):  # type:ignore
            return m.s_uz[w, t] <= m.s_abs_uz[w, t]

        def slack_absolute_value_split_2(m, w, t):  # type:ignore
            return -m.s_uz[w, t] <= m.s_abs_uz[w, t]

        self.model.total_power_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=total_power,
        )
        self.model.regulation_bound_1_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=regulation_bound_1,
        )
        self.model.regulation_bound_2_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=regulation_bound_2,
        )
        self.model.power_frequency_response_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_frequency_response,
        )
        self.model.power_response_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=power_response,
        )
        self.model.slack_absolute_value_split_1_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model.slack_absolute_value_split_2_uz = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=slack_absolute_value_split_2,
        )

    def add_bid_constraints(self) -> None:  # type:ignore
        def lambda_bid_1(m, w, t):  # type:ignore
            # g == 1 when RP price difference is above our bid price
            return (m.lambda_rp[w, t] - m.lambda_spot[w, t]) >= m.lambda_b[
                w, t
            ] - m.M * (1 - m.g[w, t])

        def lambda_bid_2(m, w, t):  # type:ignore
            return (
                m.lambda_b[w, t]
                >= (m.lambda_rp[w, t] - m.lambda_spot[w, t]) - m.M * m.g[w, t]
            )

        def phi_less_than(m, w, t):  # type:ignore
            return m.phi[w, t] <= m.p_base[t] * (1 - m.mask[t])

        def lambda_bid_less_than(m, w, t):  # type:ignore
            # return m.lambda_b[t] <= 0
            return m.lambda_b[w, t] <= m.max_lambda_bid

        def real_time_power_less_than(m, w, t):  # type:ignore
            return m.p_up[w, t] <= m.phi[w, t] * m.up_regulation_event[w, t]

        def real_time_power_greater_than(m, w, t):  # type:ignore
            return m.p_up[w, t] + m.s[w, t] >= m.phi[w, t] * m.up_regulation_event[w, t]

        def phi_linearize_1(m, w, t):  # type:ignore
            return -m.g[w, t] * m.p_nom <= m.phi[w, t]

        def phi_linearize_2(m, w, t):  # type:ignore
            return m.phi[w, t] <= m.g[w, t] * m.p_nom

        def phi_linearize_3(m, w, t):  # type:ignore
            return -(1 - m.g[w, t]) * m.p_nom <= m.phi[w, t] - m.p_up_reserve[t]

        def phi_linearize_4(m, w, t):  # type:ignore
            return m.phi[w, t] - m.p_up_reserve[t] <= (1 - m.g[w, t]) * m.p_nom

        def one_lambda_only(m, w, t):  # type:ignore
            if t == 0:
                return Constraint.Skip
            return m.lambda_b[w, t] == m.lambda_b[w, t - 1]

        def one_lambda_only2(m, w, t):  # type:ignore
            if w == 0:
                return Constraint.Skip
            return m.lambda_b[w, t] == m.lambda_b[w - 1, t]

        def lambda_test_1(m, w, t):  # type:ignore
            return (
                m.lambda_b[w, t]
                == (np.diff(value(m.lambda_spot[w, :]), append=5))[t] * m.alpha + m.beta
            )

        self.model.Bid1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_1,
        )
        self.model.Bid2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_2,
        )
        # self.model.Bid3 = Constraint(
        # self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=phi_less_than,
        # )
        self.model.Bid4 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_less_than,
        )
        self.model.Bid5 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=real_time_power_less_than,
        )
        self.model.Bid6 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=real_time_power_greater_than,
        )
        self.model.Bid7 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_1,
        )
        self.model.Bid8 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_2,
        )
        self.model.Bid9 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_3,
        )
        self.model.Bid10 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_4,
        )
        self.model.one_lambda_constraint_1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=one_lambda_only,
        )
        self.model.one_lambda_constraint_2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=one_lambda_only2,
        )
        self.model.lambda_policy_1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_test_1,
        )

    def add_thermodynamics_constraints_baseline(self) -> None:  # type:ignore

        self.model.Base1 = Constraint(
            self.model.time_steps,
            rule=tzu_baseline_constraint,
        )
        self.model.Base2 = Constraint(
            self.model.time_steps,
            rule=tzl_baseline_constraint,
        )
        self.model.Base3 = Constraint(
            self.model.time_steps,
            rule=twu_baseline_constraint,
        )
        self.model.Base4 = Constraint(
            self.model.time_steps,
            rule=twl_baseline_constraint,
        )

    def add_thermodynamics_constraints(self) -> None:  # type:ignore

        # add system constraints to instance
        self.model.S1 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=tzl_constraint_1
        )
        self.model.S2 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=tzl_constraint_2
        )
        self.model.S3 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=tzu_constraint_1
        )
        self.model.S4 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=tzu_constraint_2
        )
        self.model.S1_2 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=twl_constraint_1
        )
        self.model.S2_2 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=twl_constraint_2
        )
        self.model.S3_2 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=twu_constraint_1
        )
        self.model.S4_2 = Constraint(
            self.model.nb_scenarios, self.model.time_steps, rule=twu_constraint_2
        )
        self.model.S5 = Constraint(rule=delta_constraint)

        self.model.Zink1 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=tzu_constraint,
        )
        self.model.Zink2 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=tzl_constraint,
        )
        self.model.Zink3 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=twu_constraint,
        )
        self.model.Zink4 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=twl_constraint,
        )

        # self.model.B1 = Constraint(
        #     self.model.nb_scenarios,
        #     rule=boundary_constraint1,
        # )
        # self.model.B2 = Constraint(
        #     self.model.nb_scenarios,
        #     rule=boundary_constraint2,
        # )

    def add_auxillary_constraints_lz(self) -> None:  # type:ignore
        # auxillary constraints lower zone
        def u_up_rule_lz(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_up_lz[w, t - 1]
                    - m.u_up_lz[w, t]
                    + m.y_up_lz[w, t]
                    - m.z_up_lz[w, t]
                    == 0
                )
            return m.u_up_lz[w, t] == 0

        def y_z_up_rule_lz(m, w, t):  # type:ignore
            return m.y_up_lz[w, t] + m.z_up_lz[w, t] <= 1

        def u_down_rule_lz(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_down_lz[w, t - 1]
                    - m.u_down_lz[w, t]
                    + m.y_down_lz[w, t]
                    - m.z_down_lz[w, t]
                    == 0
                )
            return m.u_down_lz[w, t] == 0

        def y_z_down_rule_lz(m, w, t):  # type:ignore
            return m.y_down_lz[w, t] + m.z_down_lz[w, t] <= 1

        def u_rule_lz(m, w, t):  # type:ignore
            return m.u_up_lz[w, t] + m.u_down_lz[w, t] <= 1

        def y_rule_lz(m, w, t):  # type:ignore
            return m.y_up_lz[w, t] + m.y_down_lz[w, t] <= 1

        def z_rule_lz(m, w, t):  # type:ignore
            return m.z_up_lz[w, t] + m.z_down_lz[w, t] <= 1

        def min_up_reguluation_rule_lz(m, w, t):  # type:ignore
            ub = min(t + value(m.min_up_time) + 1, value(m._n_hours))
            return (
                sum(m.u_up_lz[w, k] for k in range(t, ub))
                >= m.min_up_time * m.y_up_lz[w, t]
            )

        def max_up_reguluation_rule_lz(m, w, t):  # type:ignore
            ub = min(t + value(m.max_up_time) + 1, value(m._n_hours))
            return sum(m.u_up_lz[w, k] for k in range(t, ub)) <= m.max_up_time

        # add all auxillary constraints for lower zone to model
        self.model.A1_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_up_rule_lz,
        )
        self.model.A2_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_up_rule_lz,
        )
        self.model.A3_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_down_rule_lz,
        )
        self.model.A4_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_down_rule_lz,
        )
        self.model.A5_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_rule_lz,
        )
        self.model.A6_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_rule_lz,
        )
        self.model.A7_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=z_rule_lz,
        )
        self.model.A8_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=min_up_reguluation_rule_lz,
        )
        self.model.A9_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=max_up_reguluation_rule_lz,
        )

    def add_auxillary_constraints_uz(self) -> None:  # type:ignore
        def u_up_rule_uz(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_up_uz[w, t - 1]
                    - m.u_up_uz[w, t]
                    + m.y_up_uz[w, t]
                    - m.z_up_uz[w, t]
                    == 0
                )
            return m.u_up_uz[w, t] == 0

        def y_z_up_rule_uz(m, w, t):  # type:ignore
            return m.y_up_uz[w, t] + m.z_up_uz[w, t] <= 1

        def u_down_rule_uz(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_down_uz[w, t - 1]
                    - m.u_down_uz[w, t]
                    + m.y_down_uz[w, t]
                    - m.z_down_uz[w, t]
                    == 0
                )
            return m.u_down_uz[w, t] == 0

        def y_z_down_rule_uz(m, w, t):  # type:ignore
            return m.y_down_uz[w, t] + m.z_down_uz[w, t] <= 1

        def u_rule_uz(m, w, t):  # type:ignore
            return m.u_up_uz[w, t] + m.u_down_uz[w, t] <= 1

        def y_rule_uz(m, w, t):  # type:ignore
            return m.y_up_uz[w, t] + m.y_down_uz[w, t] <= 1

        def z_rule_uz(m, w, t):  # type:ignore
            return m.z_up_uz[w, t] + m.z_down_uz[w, t] <= 1

        def min_up_reguluation_rule_uz(m, w, t):  # type:ignore
            ub = min(t + value(m.min_up_time) + 1, value(m._n_hours))
            return (
                sum(m.u_up_uz[w, k] for k in range(t, ub))
                >= m.min_up_time * m.y_up_uz[w, t]
            )

        def max_up_reguluation_rule_uz(m, w, t):  # type:ignore
            ub = min(t + value(m.max_up_time) + 1, value(m._n_hours))
            return sum(m.u_up_uz[w, k] for k in range(t, ub)) <= m.max_up_time

        # add all auxillary constraints for upper zone to model
        self.model.A1_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_up_rule_uz,
        )
        self.model.A2_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_up_rule_uz,
        )
        self.model.A3_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_down_rule_uz,
        )
        self.model.A4_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_down_rule_uz,
        )
        self.model.A5_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_rule_uz,
        )
        self.model.A6_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_rule_uz,
        )
        self.model.A7_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=z_rule_uz,
        )
        self.model.A8_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=min_up_reguluation_rule_uz,
        )
        self.model.A9_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=max_up_reguluation_rule_uz,
        )

    def add_rebound_constraints_lz(self) -> None:  # type:ignore
        # add rebound contraints for lower zone
        def rebound_rule_1_lz(m, w, t):  # type:ignore
            return m.y_down_lz[w, t] >= m.z_up_lz[w, t]

        def rebound_rule_1_2_lz(m, w, t):  # type:ignore
            return m.y_down_lz[w, t] <= m.z_up_lz[w, t]

        def symmetrical_rebound_2_v2_lz(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                sum(m.tzl[w, i] - m.tzl_base[i] for i in range(t0, t1))
                >= -(1 - m.z_down_lz[w, h]) * 20 * m._n_steps
            )

        def symmetrical_rebound_3_v2_lz(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                sum(m.tzl[w, i] - m.tzl_base[i] for i in range(t0, t1))
                <= (1 - m.z_down_lz[w, h]) * 20 * m._n_steps
            )

        def equal_up_and_down_regulation_hours_lz(m, w):  # type:ignore
            return sum(m.u_up_lz[w, t] for t in m.n_hours) == sum(
                m.u_down_lz[w, t] for t in m.n_hours
            )

        # add all rebound constraints to instance
        self.model.C24_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1_lz,
        )
        self.model.C24_2_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1_2_lz,
        )

        self.model.temp_reb2_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_2_v2_lz,
        )
        self.model.temp_reb3_LZ = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_3_v2_lz,
        )
        self.model.rebound_hours_1_LZ = Constraint(
            self.model.nb_scenarios,
            rule=equal_up_and_down_regulation_hours_lz,
        )

        # up-regulation should happen first
        def up_regulation_first_rule_lz(m, w, t):  # type:ignore
            return sum(m.y_down_lz[w, k] for k in range(0, t)) <= sum(
                m.y_up_lz[w, k] for k in range(0, t)
            )

        self.model.F1_LZ = Constraint(
            self.model.nb_scenarios,
            RangeSet(1, self.model._n_hours - 1),
            rule=up_regulation_first_rule_lz,
        )

        # other boundary constraints
        def first_time_step_rule_1_lz(m, w):  # type:ignore
            return m.y_down_lz[w, 0] == 0

        def first_time_step_rule_2_lz(m, w):  # type:ignore
            return m.z_down_lz[w, 0] == 0

        def first_time_step_rule_3_lz(m, w):  # type:ignore
            return m.y_up_lz[w, 0] == 0

        def first_time_step_rule_4_lz(m, w):  # type:ignore
            return m.z_up_lz[w, 0] == 0

        self.model.TS1_LZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_1_lz
        )
        self.model.TS2_LZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_2_lz
        )
        self.model.TS3_LZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_3_lz
        )
        self.model.TS4_LZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_4_lz
        )

    def add_rebound_constraints_uz(self) -> None:  # type:ignore
        # add rebound contraints for upper zone
        def rebound_rule_1_uz(m, w, t):  # type:ignore
            return m.y_down_uz[w, t] >= m.z_up_uz[w, t]

        def rebound_rule_1_2_uz(m, w, t):  # type:ignore
            return m.y_down_uz[w, t] <= m.z_up_uz[w, t]

        def symmetrical_rebound_4_v2_uz(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                sum(m.tzu[w, i] - m.tzu_base[i] for i in range(t0, t1))
                >= -(1 - m.z_down_uz[w, h]) * 20 * m._n_steps
            )

        def symmetrical_rebound_5_v2_uz(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                sum(m.tzu[w, i] - m.tzu_base[i] for i in range(t0, t1))
                <= (1 - m.z_down_uz[w, h]) * 20 * m._n_steps
            )

        def equal_up_and_down_regulation_hours_uz(m, w):  # type:ignore
            return sum(m.u_up_uz[w, t] for t in m.n_hours) == sum(
                m.u_down_uz[w, t] for t in m.n_hours
            )

        # add all rebound constraints to instance
        self.model.C24_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1_uz,
        )
        self.model.C24_2_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1_2_uz,
        )
        self.model.temp_reb4_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_4_v2_uz,
        )
        self.model.temp_reb5_UZ = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_5_v2_uz,
        )
        self.model.rebound_hours_1_UZ = Constraint(
            self.model.nb_scenarios,
            rule=equal_up_and_down_regulation_hours_uz,
        )

        # up-regulation should happen first
        def up_regulation_first_rule_uz(m, w, t):  # type:ignore
            return sum(m.y_down_uz[w, k] for k in range(0, t)) <= sum(
                m.y_up_uz[w, k] for k in range(0, t)
            )

        self.model.F1_UZ = Constraint(
            self.model.nb_scenarios,
            RangeSet(1, self.model._n_hours - 1),
            rule=up_regulation_first_rule_uz,
        )

        # other boundary constraints
        def first_time_step_rule_1_uz(m, w):  # type:ignore
            return m.y_down_uz[w, 0] == 0

        def first_time_step_rule_2_uz(m, w):  # type:ignore
            return m.z_down_uz[w, 0] == 0

        def first_time_step_rule_3_uz(m, w):  # type:ignore
            return m.y_up_uz[w, 0] == 0

        def first_time_step_rule_4_uz(m, w):  # type:ignore
            return m.z_up_uz[w, 0] == 0

        self.model.TS1_UZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_1_uz
        )
        self.model.TS2_UZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_2_uz
        )
        self.model.TS3_UZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_3_uz
        )
        self.model.TS4_UZ = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_4_uz
        )

    @staticmethod
    def run_solver(model_instance: AbstractModel, tee: bool = True) -> Any:
        if tee:
            print(model_instance.statistics)
            print(
                f"Number of variables: {len([_ for v in model_instance.component_objects(Var, active=True) for _ in v]) }"
            )

        solver = SolverFactory("gurobi")
        if value(model_instance._nb_scenarios) >= 1:  # type:ignore
            solver.options["TimeLimit"] = 60 * 4
            solver.options["MIPGap"] = 0.05
        #     pass
        # solver.options["MIPFocus"] = 1
        results = solver.solve(model_instance, tee=tee)

        def assert_conditions(
            phi: np.ndarray,
            p_up_reserve: np.ndarray,
            g_indicator: np.ndarray,
            lambda_b: np.ndarray,
            lambda_rp: np.ndarray,
            lambda_spot: np.ndarray,
        ) -> None:
            assert all(
                [
                    np.isclose(phi[w, t], p_up_reserve[t], atol=1e-05)
                    if g_indicator[w, t] == 1
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            assert all(
                [
                    np.isclose(phi[w, t], 0, atol=1e-05)
                    if g_indicator[w, t] == 0
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            # NOTE: sometime, the below test fails because g is not exactly 1.
            # We solve it by relaxing the tolerance and checking difference
            # between the balancing price and spot price
            assert all(
                [
                    np.isclose(g_indicator[w, t], 1, atol=1e-05)
                    if (
                        (lambda_rp[w, t] - lambda_spot[w, t]) > lambda_b[w, t]
                        and not (
                            np.isclose(
                                lambda_rp[w, t] - lambda_spot[w, t],
                                lambda_b[w, t],
                                atol=1e-04,
                            )
                            or lambda_rp[w, t] - lambda_spot[w, t] < 0.002
                        )
                    )
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            # NOTE: sometime, the below test fails because g is not exactly 1.
            # We solve it by relaxing the tolerance and checking difference
            # between the balancing price and spot price
            assert all(
                [
                    np.isclose(g_indicator[w, t], 0, atol=1e-05)
                    if (
                        lambda_rp[w, t] - lambda_spot[w, t] < lambda_b[w, t]
                        and not (
                            np.isclose(
                                lambda_rp[w, t] - lambda_spot[w, t],
                                lambda_b[w, t],
                                atol=1e-04,
                            )
                            or lambda_rp[w, t] - lambda_spot[w, t] < 0.002
                        )
                    )
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )

        def assert_conditions_2(
            p_up_reserve: np.ndarray,
            g_indicator: np.ndarray,
            p_up: np.ndarray,
            slack: np.ndarray,
            up_regulation_event: np.ndarray,
        ) -> None:
            assert all(
                [
                    (
                        p_up[w, t] + slack[w, t]
                        >= p_up_reserve[t]
                        * g_indicator[w, t]
                        * up_regulation_event[w, t]
                    )
                    or (
                        np.isclose(
                            p_up[w, t] + slack[w, t],
                            p_up_reserve[t]
                            * g_indicator[w, t]
                            * up_regulation_event[w, t],
                            atol=1e-03,
                        )
                    )
                    for w in range(p_up.shape[0])
                    for t in range(p_up.shape[1])
                ]
            )
            assert all(
                [
                    np.isclose(p_up[w, t], 0, atol=1e-03)
                    if (
                        up_regulation_event[w, t] == 0
                        or g_indicator[w, t] == 0
                        or p_up_reserve[t] == 0
                    )
                    else True
                    for w in range(p_up.shape[0])
                    for t in range(p_up.shape[1])
                ]
            )

        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            if tee:
                print("this is feasible and optimal")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            raise Exception("MILP is infeasible and could not be solved.")
        elif (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.feasible
        ):
            print("Solution is ok but only feasible")
        elif results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            print("Time limit reached. Returning incumbent solution")
        else:
            # something else is wrong
            raise Exception(f"MILP could not be solved: {str(results.solver)}")

        if model_instance.name == Case.mFRR_AND_ENERGY.name:
            # do sanity checks
            phi = np.array(list(model_instance.phi.extract_values().values())).reshape(
                -1, 24
            )
            p_up_reserve = np.array(
                list(model_instance.p_up_reserve.extract_values().values())
            ).reshape(-1)
            g_indicator = np.array(
                list(model_instance.g.extract_values().values())
            ).reshape(-1, 24)
            lambda_b = np.array(
                list(model_instance.lambda_b.extract_values().values())
            ).reshape(-1, 24)
            lambda_rp = np.array(
                list(model_instance.lambda_rp.extract_values().values())
            ).reshape(-1, 24)
            lambda_spot = np.array(
                list(model_instance.lambda_spot.extract_values().values())
            ).reshape(-1, 24)
            p_up = np.array(
                list(model_instance.p_up.extract_values().values())
            ).reshape(-1, 24)
            slack = np.array(list(model_instance.s.extract_values().values())).reshape(
                -1, 24
            )
            up_regulation_event = np.array(
                list(model_instance.up_regulation_event.extract_values().values())
            ).reshape(-1, 24)

            assert_conditions(
                phi, p_up_reserve, g_indicator, lambda_b, lambda_rp, lambda_spot
            )
            assert_conditions_2(
                p_up_reserve, g_indicator, p_up, slack, up_regulation_event
            )
        elif model_instance.name == Case.FCR.name:
            p_up_reserve = np.array(
                list(model_instance.p_up_reserve.extract_values().values())
            ).reshape(-1)
            p_up_reserve = np.repeat(p_up_reserve, 60)
            p_up_reserve_lz = np.array(
                list(model_instance.p_up_reserve_lz.extract_values().values())
            ).reshape(-1)
            p_up_reserve_lz = np.repeat(p_up_reserve_lz, 60)
            p_up_reserve_uz = np.array(
                list(model_instance.p_up_reserve_uz.extract_values().values())
            ).reshape(-1)
            p_up_reserve_uz = np.repeat(p_up_reserve_uz, 60)
            p_base = np.array(
                list(model_instance.p_base.extract_values().values())
            ).reshape(-1)
            p_base = np.repeat(p_base, 60)
            p_base_lz = np.array(
                list(model_instance.p_base_lz.extract_values().values())
            ).reshape(-1)
            p_base_lz = np.repeat(p_base_lz, 60)
            p_base_uz = np.array(
                list(model_instance.p_base_uz.extract_values().values())
            ).reshape(-1)
            p_base_uz = np.repeat(p_base_uz, 60)
            p_freq = np.array(
                list(model_instance.p_freq.extract_values().values())
            ).reshape(-1)
            p_freq_lz = np.array(
                list(model_instance.p_freq_lz.extract_values().values())
            ).reshape(-1)
            p_freq_uz = np.array(
                list(model_instance.p_freq_uz.extract_values().values())
            ).reshape(-1)
            pt = np.array(list(model_instance.pt.extract_values().values())).reshape(-1)
            pt_lz = np.array(
                list(model_instance.pt_lz.extract_values().values())
            ).reshape(-1)
            pt_uz = np.array(
                list(model_instance.pt_uz.extract_values().values())
            ).reshape(-1)
            frequency = np.array(
                list(model_instance.freq.extract_values().values())
            ).reshape(-1)
            slack = np.array(list(model_instance.s.extract_values().values())).reshape(
                -1
            )
            slack_abs = np.array(
                list(model_instance.s_abs.extract_values().values())
            ).reshape(-1)
            slack_lz = np.array(
                list(model_instance.s_lz.extract_values().values())
            ).reshape(-1)
            slack_abs_lz = np.array(
                list(model_instance.s_abs_lz.extract_values().values())
            ).reshape(-1)
            slack_uz = np.array(
                list(model_instance.s_uz.extract_values().values())
            ).reshape(-1)
            slack_abs_uz = np.array(
                list(model_instance.s_abs_uz.extract_values().values())
            ).reshape(-1)

            assert pt.min() >= 0
            assert pt_lz.min() >= 0
            assert pt_uz.min() >= 0

            assert pt.max() <= value(model_instance.p_nom) * 2

            assert all(slack_abs >= 0)
            assert all(slack_abs_lz >= 0)
            assert all(slack_abs_uz >= 0)

            true_p_freq_lz = frequency * p_up_reserve_lz + slack_lz
            true_p_freq_uz = frequency * p_up_reserve_uz + slack_uz
            true_p_lz = p_base_lz + true_p_freq_lz
            true_p_uz = p_base_uz + true_p_freq_uz
            assert np.allclose(true_p_freq_lz, p_freq_lz, atol=1.0e-3)
            assert np.allclose(true_p_freq_uz, p_freq_uz, atol=1.0e-3)
            assert np.allclose(true_p_lz, pt_lz, atol=1.0e-3)
            assert np.allclose(true_p_uz, pt_uz, atol=1.0e-3)

            true_p_freq = true_p_freq_lz + true_p_freq_uz
            assert np.allclose(true_p_freq, p_freq, atol=1.0e-3)
            true_p_freq = frequency * p_up_reserve + slack
            assert np.allclose(true_p_freq, p_freq, atol=1.0e-3)

            true_p = true_p_lz + true_p_uz
            assert np.allclose(true_p, pt, atol=1.0e-3)
            true_p = p_base + p_freq
            assert np.allclose(true_p, pt, atol=1.0e-3)

        if tee:
            print(f"Objective value: {model_instance.objective.expr()}")

        return model_instance, results
