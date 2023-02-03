from typing import Any, Dict

import numpy as np
import pandas as pd

from src.admm import ADMM
from src.base import OBJECTIVE_FUNCTION, SCENARIO_PATH, Case
from src.evaluation import evaluate_oos
from src.experiment_manager.cache import cache

# from src.fcr import FCRProblem, FCRSolverInstance
from src.prepare_problem_v2 import (
    Problem,
    build_oos_scenarios,
    get_arbitrary_scenarios,
    get_chunk_instance,
    get_scenarios_fcr,
    get_variables_and_params,
)
from src.problem_v2 import SolverInstance


@cache
def run_mfrr_spot_optmization(partition: str) -> Any:

    params: Dict[str, Any] = eval(partition)

    case: Case = Case[params["case"]]
    run_oos: bool = params["run_oos"]
    nb_scenarios_spot: int = params.get("nb_scenarios_spot", 1)

    assert run_oos

    generic_solver_instance = SolverInstance(case)

    df_scenarios = pd.read_csv(
        SCENARIO_PATH,
        parse_dates=["HourUTC"],
    ).query(f"HourUTC.dt.year == {params['year']}")
    # build uncertainty set for particular year-month combination
    # scenarios = build_uncertainty_set_v2(df_scenarios, nb=1, run_oos=run_oos)
    # scenarios = build_test_uncertainty_set(df_scenarios)
    # NOTE: in-sample scenarios are sampled arbitrarily from 2021
    # NOTE: out-of-sample scenarios are simply all days in 2022
    # TODO: get scenarios for all of 2022 (right now, only until October)
    scenarios = (
        # build_uncertainty_set_v2(df_scenarios, nb=1, run_oos=run_oos)
        get_arbitrary_scenarios(df_scenarios, nb_scenarios_spot, seed=nb_scenarios_spot)
        if not run_oos
        else build_oos_scenarios(df_scenarios)
    )
    print(f"\n\nUsing {scenarios.lambda_spot.shape[0]} scenarios")

    instance = get_chunk_instance(scenarios, case)
    if not run_oos:
        # Use robust optimization due to computational complexity
        instance.convert_to_robust_scenario()
    instance.one_lambda = params["one_lambda"]

    # Add tariffs and taxes
    instance.elafgift = params["elafgift"]
    instance.moms = params["moms"]
    instance.tariff = np.zeros(24)
    instance.delta_max = params["delta_max"]
    # Divide spot prices by 2 (Coop pays 50% spot price + a fixed price). Fixed price assumed to be 0.7 kr/kWh
    # instance.lambda_spot = scenarios.lambda_spot / 2 + 0.7 # only for tariff B

    if case.name == Case.SPOT.name:
        # old_lambda_rp = instance.lambda_rp.copy()
        # force model to always activate
        instance.lambda_rp = scenarios.lambda_spot + 10
        instance.up_regulation_event = np.ones(shape=instance.lambda_rp.shape)

    # if params["analysis"] == "analysis2":
    #     instance.one_lambda = False

    # TODO: make admm work for zinc furnace
    if params.get("admm") and case.name == Case.mFRR_AND_ENERGY.name and not run_oos:
        print("Initializing ADMM")
        assert not params["one_lambda"]
        # Solve with ADMM by decomposing scenarios
        admm = ADMM(
            generic_solver_instance,
            instance,
            OBJECTIVE_FUNCTION[case.name],
            gamma=params.get("gamma", 0.5),
        )
        runs = admm.run_admm()
        res_inst, opt_result = admm.prepare_in_sample_results(runs[-1])

        if params.get("save_admm_iterations") is not None:
            return runs, res_inst, opt_result
    else:

        if not run_oos:
            # Solve normally in-sample
            # Create model instance and set apprioriate constraints
            print("Initializing model instance")
            problem = Problem(generic_solver_instance, instance)
            Problem.customize_constraints(problem.model_instance, instance.one_lambda)
            problem.set_objective(OBJECTIVE_FUNCTION[case.name])

            # if params["analysis"] == "analysis2":
            #     # full reserveation, but allow dynamic bids
            #     fix_p_up_reserve_only(problem, instance, deepcopy(params))

            opt_result = problem.solve()
            res_inst = get_variables_and_params(problem.res_instance, case)
        else:
            res_inst, opt_result = evaluate_oos(
                generic_solver_instance, instance, case, params
            )

        # if case.name == Case.SPOT.name:
        #     instance.lambda_rp = old_lambda_rp  # needed for plots

    print(opt_result.__repr__())

    return res_inst, opt_result


@cache
def run_fcr_optmization(partition: str) -> Any:

    params: Dict[str, Any] = eval(partition)

    case: Case = Case[params["case"]]
    run_oos: bool = params["run_oos"]
    year: int = params["year"]
    # only used for in-sample training in multiple scenarios
    nb_scenarios_spot: int = params.get("nb_scenarios_spot", 1)

    assert run_oos

    generic_solver_instance = SolverInstance(case)
    scenarios = get_scenarios_fcr(nb_scenarios_spot, run_oos, year)

    print(f"\n\nUsing {scenarios.lambda_spot.shape[0]} scenarios")

    # TODO: update to use FCR prices and frequency
    instance = get_chunk_instance(scenarios, case)
    if not run_oos:
        # Use single instance to learn reservation capacity for frequency
        instance.reduce_instance()

    # Add tariffs and taxes
    instance.elafgift = params["elafgift"]
    instance.moms = params["moms"]
    instance.tariff = np.zeros(24)
    instance.delta_max = params["delta_max"]

    # TODO: use real frequency data instead of random
    # instance.frequency = np.clip(
    #     np.random.normal(0, 0.3, size=(instance.nb_scenarios, 24 * 60)), -1, 1
    # )
    # TODO: use real data for frequency in DK1
    # instance.lambda_rp = np.mean(scenarios.lambda_rp, axis=0).reshape(1, -1)

    if not run_oos:
        print("Initializing model instance")
        problem = Problem(generic_solver_instance, instance)
        problem.set_objective(OBJECTIVE_FUNCTION[case.name])
        opt_result = problem.solve()
        res_inst = get_variables_and_params(problem.res_instance, case)
    else:
        # NOTE: at the moment, we only solve FCR with full hindsight (i.e., OOS like without fixing policies)
        res_inst, opt_result = evaluate_oos(
            generic_solver_instance, instance, case, params
        )

    print(opt_result.__repr__())

    return res_inst, opt_result
