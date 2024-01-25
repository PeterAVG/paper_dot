import time
from typing import Any, Dict, List, cast

import numpy as np

from src.base import OBJECTIVE_FUNCTION, Case
from src.experiment_manager.cache import load_cache, load_tmp_cache, save_to_tmp_cache
from src.objective import _o_energy_cost
from src.prepare_problem_v2 import (
    OptimizationResult,
    Problem,
    get_variables_and_params,
    value,
)
from src.problem_v2 import OptimizationInstance, SolverInstance


def fix_model(
    problem: Problem, instance: OptimizationInstance, params: Dict[str, Any]
) -> None:

    assert params["run_oos"], "We only fix the model for out-of-sample runs"
    assert (
        params["year"] == 2022
    ), "We only fix the model for out-of-sample runs in 2022"

    params["year"] = 2021
    params["run_oos"] = False
    cache: Dict[str, Any] = load_cache()

    try:
        result = cache[params.__repr__()]
    except KeyError:
        print("File not found. Can't run OOS. Stopping.")
        raise Exception("File not found. Can't run OOS. Stopping.")

    if params.get("save_admm_iterations") is not None:
        information = result[1]
    else:
        information = result[0]

    p_up_reserve = information.p_up_reserve
    lambda_spot = instance.lambda_spot

    if params["one_lambda"]:
        lambda_b = information.lambda_b
    else:
        # TODO: harmonize below with ADMM "prepare_in_sample_results"
        alpha = information.alpha
        beta = information.beta
        lambda_bid = np.diff(lambda_spot, append=5) * alpha + beta
        # lambda_bid = np.exp(np.diff(lambda_spot, append=1)) * alpha + beta

        if (lambda_bid < 0).any():
            beta += abs(np.min(lambda_bid)) + 1e-4

        problem.model_instance.alpha.fix(alpha)
        problem.model_instance.beta.fix(beta)

    for w in range(instance.nb_scenarios):
        # set parameters for all scenarios (a given day)
        for i in range(24):
            # fix decision variables related to offering strategy
            v = (
                0
                if (np.isclose(p_up_reserve[i], 0, atol=1e-05))
                # else instance.p_base[i] # out-comment this line if p_base is not fixed
                else p_up_reserve[i]
            )
            problem.model_instance.p_up_reserve[i].fix(v)
            if params["one_lambda"]:
                # 'w' if scenario dependent
                _w = 0 if lambda_b.shape[0] == 1 else w  # type:ignore
                problem.model_instance.lambda_b[w, i].fix(
                    lambda_b[_w, i]  # type:ignore
                )


def fix_p_up_reserve_only(
    problem: Problem, instance: OptimizationInstance, params: Dict[str, Any]
) -> None:
    params["year"] = 2021
    params["run_oos"] = False
    cache: Dict[str, Any] = load_cache()

    try:
        information = cache[params.__repr__()][0]
    except FileNotFoundError:
        print("File not found. Can't run OOS. Stopping.")
        raise Exception("File not found. Can't run OOS. Stopping.")

    p_up_reserve = information.p_up_reserve
    # set parameters for all scenarios (a given day)
    for i in range(24):
        v = p_up_reserve[i]
        problem.model_instance.p_up_reserve[i].fix(v)


def average_opt_results(
    opt_results: List[OptimizationResult],
) -> OptimizationResult:
    # average results
    # NOTE: we assume equal probability for each scenario
    avg_opt_result = OptimizationResult(
        base_cost_today=cast(float, np.mean([e.base_cost_today for e in opt_results])),
        total_cost=cast(float, np.mean([e.total_cost for e in opt_results])),
        expected_energy_cost=cast(
            float, np.mean([e.expected_energy_cost for e in opt_results])
        ),
        rebound_cost=cast(float, np.mean([e.rebound_cost for e in opt_results])),
        reserve_payment=cast(float, np.mean([e.reserve_payment for e in opt_results])),
        act_payment=cast(float, np.mean([e.act_payment for e in opt_results])),
        penalty_cost=cast(float, np.mean([e.penalty_cost for e in opt_results])),
        battery_capacity=-1,
    )
    return avg_opt_result


def evaluate_oos(
    generic_solver_instance: SolverInstance,
    instance: OptimizationInstance,
    case: Case,
    params: Dict[str, Any],
) -> Any:
    # solve out-of-sample in chunks to speed up evaulation
    # TODO: perhaps parallelize this

    objective = OBJECTIVE_FUNCTION[case.name]
    # TODO: chunk size > 1 does NOT give same base cost. Something is wrong. Use 1 for now.
    chunk_size = 1

    opt_results = []
    res_instances = []

    # tmp_cache = "tmp2.pkl"
    # tmp = load_tmp_cache(tmp_cache)
    # assert len(tmp) == 0, "tmp cache should be empty"

    print(f"Preparing evaluation results with chunk size {chunk_size} ...")
    start = time.time()

    for i, chunk_instance in enumerate(instance.chunk_generator(chunk_size)):
        assert chunk_instance.nb_scenarios == 1, "We only support one scenario for OOS"
        print(f"   Evaluating chunk {i}...")

        # if i in tmp:
        #     print(
        #         f"   Chunk {i} already evaluated and is present in tmp cache. Skipping..."
        #     )
        #     res_inst, opt_result = tmp[i]
        #     opt_results.append(opt_result)
        #     res_instances.append(res_inst)
        #     continue

        # neccessary to avoid error as penalty is at 20 DKK/kWh (for mFRR)
        chunk_instance.lambda_rp = np.minimum(19, chunk_instance.lambda_rp)
        problem = Problem(generic_solver_instance, chunk_instance)
        problem.set_objective(objective)

        # NOTE: at the moment, we only solve FCR and mFRR with full hindsight (i.e., OOS like without fixing policies)
        if case.name in [Case.mFRR_AND_ENERGY.name]:
            # only fix policy for mFRR. Spot is always reoptimized, i.e., there is no bid policy
            Problem.customize_constraints(problem.model_instance, instance.one_lambda)
            # fix_model(problem, chunk_instance, deepcopy(params))
        elif case.name in [Case.FCR.name]:
            # only fix required reservation capacity
            # fix_p_up_reserve_only(problem, chunk_instance, deepcopy(params))
            pass

        try:
            opt_result = problem.solve(tee=True)
        except Exception as e:
            print(f"\n\nError in chunk {i}:\n{e}\n\n")
            opt_results.append(None)
            res_instances.append(None)
            continue

        res_inst = get_variables_and_params(problem.res_instance, case)

        if case.name == Case.SPOT.name:
            # Handle this case differently
            energy_cost = value(_o_energy_cost(problem.res_instance)) * (  # type:ignore
                -1
            )
            opt_result.expected_energy_cost = energy_cost
            opt_result.total_cost = energy_cost
            opt_result.reserve_payment = 0.0
            opt_result.act_payment = 0.0
            opt_result.penalty_cost = 0.0

        opt_results.append(opt_result)
        res_instances.append(res_inst)

        # tmp[i] = (res_inst, opt_result)
        # save_to_tmp_cache(tmp, tmp_cache)

    # delete tmp cache
    # save_to_tmp_cache({}, tmp_cache)

    print(f"Evaluation done in {time.time() - start} seconds")

    # TODO: if not equal probability for all scenarios, this is not correct
    return res_instances, opt_results  # type:ignore
