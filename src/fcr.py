from typing import Any, Callable, Dict, Optional

import cloudpickle
import numpy as np
from pyomo.environ import (
    AbstractModel,
    Constraint,
    NonNegativeReals,
    NonPositiveReals,
    Objective,
    Param,
    RangeSet,
    Reals,
    SolverFactory,
    Var,
    floor,
    maximize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition
from tex.dot_nordic.scripts.base import (
    PICKLE_FOLDER,
    FCRInstanceInformation,
    OptimizationInstanceZincFurnace,
    OptimizationResult,
)
from tex.dot_nordic.scripts.constraints import (
    delta_constraint,
    twl_baseline_constraint,
    twl_constraint,
    twu_baseline_constraint,
    twu_constraint,
    tzl_baseline_constraint,
    tzl_constraint,
    tzl_constraint_1,
    tzl_constraint_2,
    tzu_baseline_constraint,
    tzu_constraint,
    tzu_constraint_1,
    tzu_constraint_2,
)
from tex.dot_nordic.scripts.objective_functions import (
    o_expected_fcr_balance_settlement,
    o_expected_fcr_energy_consumption,
    o_expected_fcr_reserve_payment,
    o_fcr_penalty,
    o_rule_fcr_energy,
)

from src.problem_v2 import OptimizationInstance


def pyomo_get_abstract_model() -> AbstractModel:

    n_steps = 24 * 60
    n_hours = 24
    nb_scenarios = 1

    model = AbstractModel()
    model.time_steps = RangeSet(0, n_steps - 1)
    model.n_hours = RangeSet(0, n_hours - 1)
    model._nb_scenarios = Param(initialize=nb_scenarios, mutable=False)
    model.nb_scenarios = RangeSet(0, model._nb_scenarios - 1)

    model.n_steps = Param(mutable=False)
    model.hour_steps = Param(mutable=False)

    model.p_base = Param(range(n_hours), mutable=False, domain=Reals)
    model.p_base_lz = Param(range(n_hours), mutable=False, domain=Reals)
    model.p_base_uz = Param(range(n_hours), mutable=False, domain=Reals)
    model.p_nom = Param(mutable=False)
    model.p_min = Param(mutable=False)

    model.ta = Param(mutable=False)
    model.Czu = Param(mutable=False)
    model.Czl = Param(mutable=False)
    model.Cwu = Param(mutable=False)
    model.Cwl = Param(mutable=False)
    model.Rww = Param(mutable=False)
    model.Rwz1 = Param(mutable=False)
    model.Rwz2 = Param(mutable=False)
    model.Rzuzl = Param(mutable=False)
    model.Rwua1 = Param(mutable=False)
    model.Rwua2 = Param(mutable=False)
    model.Rwla1 = Param(mutable=False)
    model.Rwla2 = Param(mutable=False)
    model.setpoint_lz = Param(mutable=False)
    model.setpoint_uz = Param(mutable=False)
    model.regime = Param(range(n_steps), mutable=False, domain=Reals)

    # base temperature data. Could probably replace base variables with this
    model.tzl_data = Param(range(n_steps), mutable=False, domain=Reals)
    model.tzu_data = Param(range(n_steps), mutable=False, domain=Reals)

    # variables for thermodynamics
    model.tzl = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.tzu = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.tzl_base = Var(range(n_steps), domain=Reals)
    model.tzu_base = Var(range(n_steps), domain=Reals)
    model.twl = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.twu = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.twl_base = Var(range(n_steps), domain=Reals)
    model.twu_base = Var(range(n_steps), domain=Reals)

    # temperature deviation
    model.delta = Var(domain=NonNegativeReals)

    model.dt = Param(mutable=False)
    model.delta_max = Param(mutable=False)

    model.max_up_time = Param(mutable=False)
    model.min_up_time = Param(mutable=False)

    model.lambda_spot = Param(
        range(nb_scenarios), range(n_hours), mutable=False, domain=Reals
    )
    model.lambda_rp = Param(
        range(nb_scenarios), range(n_hours), mutable=False, domain=Reals
    )
    model.lambda_fcr = Param(
        range(nb_scenarios), range(n_hours), mutable=False, domain=Reals
    )
    model.probabilities = Param(range(nb_scenarios), mutable=False)

    model.elafgift = Param(mutable=False)
    model.moms = Param(mutable=False)
    model.tariff = Param(range(n_hours), mutable=False, domain=Reals)

    # reservation capacity (up mFRR)
    model.p_up_reserve_lz = Var(range(n_hours), domain=NonNegativeReals)
    model.p_up_reserve_uz = Var(range(n_hours), domain=NonNegativeReals)
    model.p_up_reserve = Var(range(n_hours), domain=NonNegativeReals)

    # power consumption
    model.pt_lz = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)
    model.pt_uz = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)
    model.pt = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)

    # power adjustment to frequency deviation
    model.p_freq_lz = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.p_freq_uz = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.p_freq = Var(range(nb_scenarios), range(n_steps), domain=Reals)

    # slack used for penalty
    model.s_lz = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.s_uz = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    model.s = Var(range(nb_scenarios), range(n_steps), domain=Reals)
    # for absolute value
    model.s_abs_lz = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)
    model.s_abs_uz = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)
    model.s_abs = Var(range(nb_scenarios), range(n_steps), domain=NonNegativeReals)

    # frequency
    model.freq = Param(range(nb_scenarios), range(n_steps), mutable=False)

    return model


class FCRSolverInstance:
    def __init__(
        self,
    ) -> None:
        self.model = pyomo_get_abstract_model()
        # add constraints to instance
        self.add_total_power_constraints()
        self.add_power_constraints_lz()
        self.add_power_constraints_uz()
        self.add_thermodynamic_constraints_baseline()
        self.add_thermodynamic_constraints()

    @staticmethod
    def instance_to_dict(inst: OptimizationInstance) -> Dict:
        data: Dict[None, Any] = {None: {}}
        data[None]["p_nom"] = {None: inst.p_nom}
        data[None]["p_min"] = {None: inst.p_min}

        # TCL specific
        data[None]["ta"] = {None: inst.ta}
        data[None]["Czu"] = {None: inst.Czu}
        data[None]["Czl"] = {None: inst.Czl}
        data[None]["Cwu"] = {None: inst.Cwu}
        data[None]["Cwl"] = {None: inst.Cwl}
        data[None]["Rww"] = {None: inst.Rww}
        data[None]["Rwz1"] = {None: inst.Rwz1}
        data[None]["Rwz2"] = {None: inst.Rwz2}
        data[None]["Rzuzl"] = {None: inst.Rzuzl}
        data[None]["Rwua1"] = {None: inst.Rwua1}
        data[None]["Rwua2"] = {None: inst.Rwua2}
        data[None]["Rwla1"] = {None: inst.Rwla1}
        data[None]["Rwla2"] = {None: inst.Rwla2}
        data[None]["setpoint_lz"] = {None: inst.setpoint_lz}
        data[None]["setpoint_uz"] = {None: inst.setpoint_uz}
        data[None]["regime"] = {i: e for i, e in enumerate(inst.regime)}

        data[None]["p_base_lz"] = {i: e for i, e in enumerate(inst.p_base_lz)}
        data[None]["p_base_uz"] = {i: e for i, e in enumerate(inst.p_base_uz)}
        data[None]["p_base"] = {i: e for i, e in enumerate(inst.p_base)}

        data[None]["tzl_data"] = {i: e for i, e in enumerate(inst.tzl_data)}
        data[None]["tzu_data"] = {i: e for i, e in enumerate(inst.tzu_data)}

        data[None]["dt"] = {None: inst.dt}
        data[None]["delta_max"] = {None: inst.delta_max}

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
        data[None]["lambda_fcr"] = {
            (i, j): inst.lambda_fcr[i, j]
            for i in range(inst.lambda_fcr.shape[0])
            for j in range(inst.lambda_fcr.shape[1])
        }
        data[None]["freq"] = {
            (i, j): inst.frequency[i, j]
            for i in range(inst.frequency.shape[0])
            for j in range(inst.frequency.shape[1])
        }

        data[None]["elafgift"] = {None: inst.elafgift}
        data[None]["moms"] = {None: inst.moms}
        data[None]["tariff"] = {i: e for i, e in enumerate(inst.tariff)}

        data[None]["probabilities"] = {i: e for i, e in enumerate(inst.probabilities)}

        data[None]["n_steps"] = {None: self.n_steps}
        data[None]["hour_steps"] = {None: self.hour_steps}

        return data

    def add_total_power_constraints(self) -> None:
        def total_reservation(m, h):
            return m.p_up_reserve[h] == m.p_up_reserve_lz[h] + m.p_up_reserve_uz[h]

        def reservation_1(m, h):
            return m.p_up_reserve[h] <= m.p_base[h]

        def total_regulation(m, w, t):
            return m.p_freq[w, t] == m.p_freq_lz[w, t] + m.p_freq_uz[w, t]

        def total_power(m, w, t):
            return m.pt[w, t] == m.pt_lz[w, t] + m.pt_uz[w, t]

        def total_slack(m, w, t):
            return m.s[w, t] == m.s_abs_lz[w, t] + m.s_abs_uz[w, t]

        def power_frequency_response(m, w, t):
            h = floor(t / m.hour_steps)
            return m.p_freq[w, t] == m.freq[w, t] * m.p_up_reserve[h] + m.s[w, t]

        def power_response(m, w, t):
            h = floor(t / m.hour_steps)
            return m.pt[w, t] == m.p_freq[w, t] + m.p_base[h]

        def slack_absolute_value_split_1(m, w, t):
            return m.s[w, t] <= m.s_abs[w, t]

        def slack_absolute_value_split_2(m, w, t):
            return -m.s[w, t] <= m.s_abs[w, t]

        def test(m, h):
            return m.p_up_reserve[h] == m.p_base[h]

        # add constraints to model
        self.model_instance.total_reservation = Constraint(
            self.model_instance.n_hours,
            rule=total_reservation,
        )
        self.model_instance.reservation_1 = Constraint(
            self.model_instance.n_hours,
            rule=reservation_1,
        )
        self.model_instance.total_up_regulation = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=total_regulation,
        )
        self.model_instance.total_power = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=total_power,
        )
        self.model_instance.power_response = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_response,
        )
        self.model_instance.power_frequency_response = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_frequency_response,
        )
        self.model_instance.total_slack = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=total_slack,
        )
        self.model_instance.slack_absolute_value_split_1 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model_instance.slack_absolute_value_split_2 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_2,
        )
        # self.model_instance.test = Constraint(
        #     rule=test,
        # )

    def add_power_constraints_lz(self) -> None:
        def total_power(m, w, t):
            h = floor(t / m.hour_steps)
            return m.pt_lz[w, t] == m.p_base_lz[h] + m.p_freq_lz[w, t]

        def regulation_bound_1(m, w, t):
            h = floor(t / m.hour_steps)
            return m.p_freq_lz[w, t] <= m.p_nom - m.p_base_lz[h]

        def regulation_bound_2(m, w, t):
            h = floor(t / m.hour_steps)
            return m.p_freq_lz[w, t] >= -m.p_base_lz[h]

        def power_frequency_response(m, w, t):
            h = floor(t / m.hour_steps)
            return (
                m.p_freq_lz[w, t] == m.freq[w, t] * m.p_up_reserve_lz[h] + m.s_lz[w, t]
            )

        def power_response(m, w, t):
            h = floor(t / m.hour_steps)
            return m.pt_lz[w, t] == m.p_freq_lz[w, t] + m.p_base_lz[h]

        def slack_absolute_value_split_1(m, w, t):
            return m.s_lz[w, t] <= m.s_abs_lz[w, t]

        def slack_absolute_value_split_2(m, w, t):
            return -m.s_lz[w, t] <= m.s_abs_lz[w, t]

        self.model_instance.total_power_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=total_power,
        )
        self.model_instance.regulation_bound_1_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=regulation_bound_1,
        )
        self.model_instance.regulation_bound_2_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=regulation_bound_2,
        )
        self.model_instance.power_frequency_response_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_frequency_response,
        )
        self.model_instance.power_response_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_response,
        )
        self.model_instance.slack_absolute_value_split_1_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model_instance.slack_absolute_value_split_2_lz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_2,
        )

    def add_power_constraints_uz(self) -> None:
        def total_power(m, w, t):
            h = floor(t / m.hour_steps)
            return m.pt_uz[w, t] == m.p_base_uz[h] + m.p_freq_uz[w, t]

        def regulation_bound_1(m, w, t):
            h = floor(t / m.hour_steps)
            return m.p_freq_uz[w, t] <= m.p_nom - m.p_base_uz[h]

        def regulation_bound_2(m, w, t):
            h = floor(t / m.hour_steps)
            return m.p_freq_uz[w, t] >= -m.p_base_uz[h]

        def power_frequency_response(m, w, t):
            h = floor(t / m.hour_steps)
            return (
                m.p_freq_uz[w, t] == m.freq[w, t] * m.p_up_reserve_uz[h] + m.s_uz[w, t]
            )

        def power_response(m, w, t):
            h = floor(t / m.hour_steps)
            return m.pt_uz[w, t] == m.p_freq_uz[w, t] + m.p_base_uz[h]

        def slack_absolute_value_split_1(m, w, t):
            return m.s_uz[w, t] <= m.s_abs_uz[w, t]

        def slack_absolute_value_split_2(m, w, t):
            return -m.s_uz[w, t] <= m.s_abs_uz[w, t]

        self.model_instance.total_power_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=total_power,
        )
        self.model_instance.regulation_bound_1_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=regulation_bound_1,
        )
        self.model_instance.regulation_bound_2_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=regulation_bound_2,
        )
        self.model_instance.power_frequency_response_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_frequency_response,
        )
        self.model_instance.power_response_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=power_response,
        )
        self.model_instance.slack_absolute_value_split_1_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_1,
        )
        self.model_instance.slack_absolute_value_split_2_uz = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=slack_absolute_value_split_2,
        )

    def add_thermodynamic_constraints_baseline(self) -> None:
        self.model_instance.S1 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzl_constraint_1,
        )
        self.model_instance.S2 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzl_constraint_2,
        )
        self.model_instance.S3 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzu_constraint_1,
        )
        self.model_instance.S4 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzu_constraint_2,
        )
        self.model_instance.Base1 = Constraint(
            self.model_instance.time_steps,
            rule=tzu_baseline_constraint,
        )
        self.model_instance.Base2 = Constraint(
            self.model_instance.time_steps,
            rule=tzl_baseline_constraint,
        )
        self.model_instance.Base3 = Constraint(
            self.model_instance.time_steps,
            rule=twu_baseline_constraint,
        )
        self.model_instance.Base4 = Constraint(
            self.model_instance.time_steps,
            rule=twl_baseline_constraint,
        )

    def add_thermodynamic_constraints(self) -> None:
        # add system constraints to instance
        self.model_instance.S5 = Constraint(rule=delta_constraint)

        self.model_instance.Zink1 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzu_constraint,
        )
        self.model_instance.Zink2 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=tzl_constraint,
        )
        self.model_instance.Zink3 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=twu_constraint,
        )
        self.model_instance.Zink4 = Constraint(
            self.model_instance.nb_scenarios,
            self.model_instance.time_steps,
            rule=twl_constraint,
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
            solver.options["MIPGap"] = 0.025
        #     pass
        # solver.options["MIPFocus"] = 1
        results = solver.solve(model_instance, tee=tee)

        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
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

        print(f"Objective value: {model_instance.objective.expr()}")

        return model_instance, results


def get_variables_and_params(instance: AbstractModel) -> FCRInstanceInformation:
    # extract solver results
    p_up_reserve = np.array(list(instance.p_up_reserve.extract_values().values()))
    p_up_reserve_lz = np.array(list(instance.p_up_reserve_lz.extract_values().values()))
    p_up_reserve_uz = np.array(list(instance.p_up_reserve_uz.extract_values().values()))
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
    p_freq_lz = np.array(list(instance.p_freq_lz.extract_values().values())).reshape(
        -1, 24 * 60
    )
    p_freq_uz = np.array(list(instance.p_freq_uz.extract_values().values())).reshape(
        -1, 24 * 60
    )

    # TCL specific
    tzl = np.array(list(instance.tzl.extract_values().values())).reshape(-1, 24 * 60)
    tzl_base = np.array(list(instance.tzl_base.extract_values().values())).reshape(-1)
    tzu = np.array(list(instance.tzu.extract_values().values())).reshape(-1, 24 * 60)
    tzu_base = np.array(list(instance.tzu_base.extract_values().values())).reshape(-1)
    twl = np.array(list(instance.twl.extract_values().values())).reshape(-1, 24 * 60)
    twl_base = np.array(list(instance.twl_base.extract_values().values())).reshape(-1)
    twu = np.array(list(instance.twu.extract_values().values())).reshape(-1, 24 * 60)
    twu_base = np.array(list(instance.twu_base.extract_values().values())).reshape(-1)

    lambda_spot = np.array(
        list(instance.lambda_spot.extract_values().values())
    ).reshape(-1, 24)

    s = np.array(list(instance.s.extract_values().values())).reshape(-1, 24 * 60)
    s_lz = np.array(list(instance.s_lz.extract_values().values())).reshape(-1, 24 * 60)
    s_uz = np.array(list(instance.s_uz.extract_values().values())).reshape(-1, 24 * 60)

    lambda_rp = np.array(list(instance.lambda_rp.extract_values().values())).reshape(
        -1, 24
    )
    lambda_fcr = np.array(list(instance.lambda_fcr.extract_values().values())).reshape(
        -1, 24
    )
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
        frequency=frequency,
        lambda_rp=lambda_rp,
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
    )


class FCRProblem:
    def __init__(
        self,
        abstract_model_instance: FCRSolverInstance,
        instance: OptimizationInstance,
    ) -> None:
        data = FCRSolverInstance.instance_to_dict(instance)
        self.model_instance = abstract_model_instance.model.create_instance(data=data)
        self.res_instance: Optional[Any] = None

    def set_objective(self, objective_function: Callable) -> None:
        self.model_instance.objective = Objective(
            rule=objective_function, sense=maximize
        )

    def solve(self, tee: bool = True) -> OptimizationResult:
        self.res_instance, _ = FCRSolverInstance.run_solver(
            self.model_instance, tee=tee
        )
        opt_result = self.get_result(self.res_instance, if_print=tee)
        return opt_result

    # def initialize_solver_instance(
    #     self,
    #     instance: OptimizationInstanceZincFurnace,
    #     objective_function: Callable,
    # ) -> None:
    #     solver_instance = FCRSolverInstance(self.model, instance)
    #     # add constraints to instance
    #     solver_instance.add_total_power_constraints()
    #     solver_instance.add_power_constraints_lz()
    #     solver_instance.add_power_constraints_uz()
    #     solver_instance.add_thermodynamic_constraints_baseline()
    #     solver_instance.add_thermodynamic_constraints()

    #     solver_instance.model_instance.objective = Objective(
    #         rule=objective_function, sense=maximize
    #     )

    #     self.solver_instance = solver_instance
    #     self.res_instance: Optional[Any] = None

    @staticmethod
    def get_result(
        res_instance: AbstractModel, multiplier: float = 1, if_print: bool = True
    ) -> OptimizationResult:

        p_base = np.array([value(res_instance.p_base[i]) for i in range(24)])
        reserve_payment = (
            value(o_expected_fcr_reserve_payment(res_instance)) * multiplier
        )
        act_payment = 0
        expected_energy_cost = (
            value(o_expected_fcr_balance_settlement(res_instance)) * multiplier
        )
        expected_power_usage = (
            value(o_expected_fcr_energy_consumption(res_instance)) * multiplier
        ) / 1000  # mwh
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
        penalty_cost = value(o_fcr_penalty(res_instance)) * multiplier

        if if_print:
            # print out statistics on earnings/cost
            print(f"Yearly earnings from mFRR reserve: {round(reserve_payment)} DKK")
            print(f"Yearly earnings from mFRR activation: {round(act_payment)} DKK")
            print(
                f"Yearly earnings from mFRR: {round(reserve_payment + act_payment)} DKK"
            )
            print(f"Base energy usage: {round(base_power_usage, 2)} MWH")
            print(f"Expected energy usage: {round(expected_power_usage, 2)} MWH")
            print(f"Base energy costs today: {round(base_cost_today)} DKK")
            print(f"Expected energy costs: {round(expected_energy_cost)} DKK")
            print(f"Expected total costs: {round(total_cost)} DKK")
            print(f"Expected penalty costs: {round(penalty_cost)} DKK")

            p_up_reserve = [
                round(e, 1) for e in res_instance.p_up_reserve.extract_values().values()
            ]
            print(f"p_up reserve policy: {p_up_reserve} kWh")

        return OptimizationResult(
            reserve_payment=reserve_payment,
            act_payment=act_payment,
            expected_energy_cost=expected_energy_cost,
            total_cost=total_cost,
            base_cost_today=base_cost_today,
            penalty_cost=penalty_cost,
            battery_capacity=0,
        )
