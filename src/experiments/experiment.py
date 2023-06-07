import logging
from typing import Any

from src.base import Case
from src.optimization import run_fcr_optmization, run_mfrr_spot_optmization

from ..experiment_manager.cache import cache  # noqa
from ..experiment_manager.core import ETLComponent

logger = logging.getLogger(__name__)


class Experiment(ETLComponent):
    def experiment_run_mfrr(self, **kwargs: Any) -> None:
        # run_oos = [False, True]
        # year = [2021, 2022]
        run_oos = [True]
        year = [2022]
        for _run_oos, _year in zip(run_oos, year):
            # temperature_deltas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
            temperature_deltas = [1, 2, 3, 4, 5, 6, 7, 10, 50]
            for delta_max in temperature_deltas:
                params = {
                    "elafgift": 0.0,
                    "moms": 0.0,
                    "year": _year,
                    "delta_max": delta_max,
                    "one_lambda": False,
                    "analysis": "analysis1",
                    "run_oos": _run_oos,
                }
                case = Case.mFRR_AND_ENERGY
                params["case"] = case.name

                partition = params.__repr__()
                logger.info(partition, kwargs)

                run_mfrr_spot_optmization(partition, **kwargs)

    def experiment_run_fcr(self, **kwargs: Any) -> None:
        run_oos = [True]
        year = [2022]
        temperature_deltas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _run_oos, _year in zip(run_oos, year):
            for delta_max in temperature_deltas:
                params = {
                    "elafgift": 0.0,
                    "moms": 0.0,
                    "year": _year,
                    "delta_max": delta_max,
                    "analysis": "analysis1",
                    "run_oos": _run_oos,
                }
                case = Case.FCR
                params["case"] = case.name

                partition = params.__repr__()
                logger.info(partition, kwargs)

                run_fcr_optmization(partition, **kwargs)
