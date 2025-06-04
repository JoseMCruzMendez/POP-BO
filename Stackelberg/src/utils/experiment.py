import jax
from typing import Tuple
from chex import PRNGKey

from src.environments.Domain.DiscreteDomain import DiscreteDomain
from src.bandits.BaseEstimator import BaseEstimator, EstimatorParams
from src.bandits.EmpiricalMean import EmpiricalMean
from src.bandits.LGPUCB import LGPUCB
from src.bandits.GPRegressor import GPRegressor
from src.bandits.LogisticUCB1 import LogisticUCB1
from src.bandits.Ecolog import Ecolog
from src.utils.globals import KERNELS
from src.utils.kernels import DuellingWrapper


def initialize_estimator(
    rng: PRNGKey,
    config: dict,
    estimator_config: dict,
    discrete_domain: DiscreteDomain,
    duelling: bool = False,
) -> Tuple[BaseEstimator, EstimatorParams]:
    if estimator_config["name"] == "EmpiricalMean":
        estimator = EmpiricalMean.create(domain=discrete_domain, duelling=duelling)
        estimator_params_update = {"delta": estimator_config["delta"]}
    elif estimator_config["name"] == "LogisticUCB1":
        estimator = LogisticUCB1.create(
            domain=discrete_domain,
            param_norm_ub=config["utility_function_params"]["param_norm_ub"],
            arm_norm_ub=estimator_config["arm_norm_ub"],
            horizon=estimator_config["num_iter"],
        )
        estimator_params_update = {
            "failure_level": estimator_config["delta"],
        }
    elif estimator_config["name"] == "Ecolog":
        estimator = Ecolog.create(
            domain=discrete_domain,
            param_norm_ub=config["utility_function_params"]["param_norm_ub"],
            arm_norm_ub=config["domain"]["norm_ub"],
            horizon=estimator_config["num_iter"],
        )
        estimator_params_update = {"failure_level": estimator_config["delta"]}
    elif estimator_config["name"] in ["LGPUCB", "GPRegressor"]:
        kernel = KERNELS[estimator_config["kernel"]].from_dict(
            estimator_config["kernel_params"]
        )
        if duelling:
            kernel = DuellingWrapper(kernel)
        if estimator_config["name"] == "LGPUCB":
            estimator = LGPUCB.create(
                domain=discrete_domain,
                rkhs_norm_ub=estimator_config["rkhs_norm_ub"],
                horizon=estimator_config["num_iter"]
                if "num_iter" in estimator_config
                else config["num_iter"],
                solver=estimator_config["solver"],
                duelling=duelling,
            )
            estimator_params_update = {
                "lambda_": estimator_config["lambda_"],
                "beta": estimator_config["beta"],
                "delta": estimator_config["delta"],
                "kernel": kernel,
                "nll_regularization_penalty": estimator_config["nll_regularization_penalty"],
            }
        elif estimator_config["name"] == "GPRegressor":
            estimator = GPRegressor.create(
                domain=discrete_domain,
                horizon=estimator_config["num_iter"],
                duelling=duelling,
            )
            estimator_params_update = {
                "lambda_": estimator_config["lambda_"],
                "beta": estimator_config["beta"],
                "delta": estimator_config["delta"],
                "kernel": kernel,
            }
    else:
        raise ValueError("Estimator not recognized.")

    estimator_params = estimator.default_params
    estimator_params = estimator.reset(rng, estimator_params)
    estimator_params = estimator_params.replace(**estimator_params_update)
    return estimator, estimator_params
