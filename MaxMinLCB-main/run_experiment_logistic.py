import pickle
import os
import yaml
import argparse
import time
import jax
from chex import PRNGKey
import jax.numpy as jnp
from typing import Dict, Tuple

from src.environments.Domain.DiscreteDomain import DiscreteDomain
from src.environments.Domain import domain_feature_generator
from src.environments.LogisticEnvironment.LogisticBandit import (
    UtilityLogisticBanditEnvironment,
    LogisticEnvParams,
)
from src.bandits.EmpiricalMean import EmpiricalMean
from src.bandits.LogisticUCB1 import LogisticUCB1
from src.bandits.LGPUCB import LGPUCB
from src.bandits.GPRegressor import GPRegressor
from src.utils.utility_functions import (
    create_linear_utility,
    create_polynomial_utility,
    create_yelp_utility,
    create_standard_optimisation_function
)
from src.utils.experiment import initialize_estimator

ALGORITHMS = {
    "EmpiricalMean": EmpiricalMean,
    "LogisticUCB1": LogisticUCB1,
    "LGPUCB": LGPUCB,
    "GPRegressor": GPRegressor,
}
UTILITY_FUNCTIONS = {
    "linear": create_linear_utility,
    "polynomial": create_polynomial_utility,
    "yelp": create_yelp_utility,
    "ackley": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "ackley"
    ),
    "hoelder": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "hoelder"
    ),
    "eggholder": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "eggholder"
    ),
    "rosenbrock": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "rosenbrock"
    ),
    "bukin": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "bukin"
    ),
    "branin": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "branin"
    ),
    "michalewicz": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "michalewicz"
    ),
    "matyas": lambda rng, domain, config: create_standard_optimisation_function(
        rng, domain, config, "matyas"
    ),
}


def initialize_environment(
    rng: PRNGKey, config: Dict
) -> Tuple[DiscreteDomain, UtilityLogisticBanditEnvironment, LogisticEnvParams]:
    # Initialize domain
    rng, _rng = jax.random.split(rng)
    if config["domain"]["initialization"] == "normal":
        arm_features = domain_feature_generator.normal(_rng, config)
    elif config["domain"]["initialization"] == "uniform":
        arm_features = domain_feature_generator.uniform(_rng, config)
    elif config["domain"]["initialization"] == "meshgrid":
        arm_features = domain_feature_generator.meshgrid(
            _rng,
            jnp.array(config["domain"]["params"]["range"]),
            int(config["num_arms"] ** (1 / config["feature_dim"])),
            config["feature_dim"],
        )
    else:
        raise ValueError(
            "Invalid arm_initialization. Use 'normal' or 'uniform'."
        )
    discrete_domain = DiscreteDomain.create(
        num_elements=config["num_arms"],
        features=arm_features,
    )
    # Initialize environment
    rng, _rng = jax.random.split(rng)
    utility_params, utility_function = UTILITY_FUNCTIONS[config["utility_function"]](
        _rng,
        discrete_domain,
        config["utility_function_params"],
    )
    env = UtilityLogisticBanditEnvironment(
        domain=discrete_domain,
        utility_function=utility_function,
    )
    env_params = env.default_params
    env_params = env_params.replace(utility_function_params=utility_params)
    best_arm, best_arm_value = env.best_arm(env_params)
    env_params = env_params.replace(best_arm=best_arm, best_arm_value=best_arm_value)
    return discrete_domain, env, env_params


def run_experiment(
    rng: PRNGKey,
    config: Dict,
    estimator_config: Dict,
    num_iter: int,
):
    """
    Run the experiment for the given environment and estimator.
    :param rng: Random key
    :param estimator_params_update: Dictionary with estimator parameters unique to the experiment
    :return:
    """
    # Initialize the environment
    discrete_domain, env, env_params = initialize_environment(rng, config)
    # Initialize the estimator
    rng, _rng = jax.random.split(rng)
    estimator, estimator_params = initialize_estimator(
        _rng, config, estimator_config, discrete_domain
    )

    # Create function to be used with jax.lax.scan for the loop
    def loop_body(carry, _):
        key, estimator_params_carry = carry
        key, _key = jax.random.split(key)
        (
            next_arm,
            posterior_mean,
            posterior_var,
            estimator_params_carry,
        ) = estimator.best_arm(_key, estimator_params_carry)
        key, _key = jax.random.split(key)
        reward = env.pull(_key, next_arm, env_params)
        regret = env.regret(next_arm, env_params)
        key, _key = jax.random.split(key)
        estimator_params_carry, update_info = estimator.update(
            _key, next_arm, reward, estimator_params_carry
        )
        logs = {
            "selected_arm": next_arm,
            "reward": reward,
            "regret": regret,
        }
        if "posteriors" in config["logging"]:
            logs["posterior_mean"] = posterior_mean
            logs["posterior_var"] = posterior_var
        if "update_info" in config["logging"]:
            for k, v in update_info.items():
                logs[k] = v
        return (key, estimator_params_carry), logs

    # Run the loop
    carry, outputs = jax.lax.scan(
        loop_body, (rng, estimator_params), None, length=num_iter
    )
    return carry, outputs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run logistic bandit experiment.")
    parser.add_argument(
        "--dir", type=str, help="Path to the experiment's directory.", required=True
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Name of the algorithm to run.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    print("Output directory: ", args.dir)
    print("Device used: ", jax.devices())

    # Load configuration
    config_path = os.path.join(args.dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration: ", config)
    rng = jax.random.PRNGKey(config["seed"])

    # Run experiment for each estimator
    for estimator_config in config["algorithms"]:
        if args.algo is not None and estimator_config["name"] != args.algo:
            continue
        print("--- Running ", estimator_config["name"], " ---")
        start_time = time.time()
        if config["grid_search"]:
            def grid_search_run(key, penalty, rkhs_norm_ub, variance, length_scale, arm_norm_ub):
                estimator_config_tmp = estimator_config.copy()
                if estimator_config["name"] in ["LGPUCB", "GPRegressor"]:
                    estimator_config_tmp["nll_regularization_penalty"] = penalty
                    estimator_config_tmp["rkhs_norm_ub"] = rkhs_norm_ub
                    estimator_config_tmp["kernel_params"]["variance"] = variance
                    estimator_config_tmp["kernel_params"]["length_scale"] = length_scale
                elif estimator_config["name"] in ["LogisticUCB1"]:
                    estimator_config_tmp["arm_norm_ub"] = arm_norm_ub
                return run_experiment(
                    key, config, estimator_config_tmp, estimator_config_tmp["num_iter"]
                )

            # Apply vmap reverse and jit
            grid_search_run_vmap = grid_search_run
            for i in range(5, -1, -1):
                in_axes = [None, None, None, None, None, None]
                in_axes[i] = 0
                grid_search_run_vmap = jax.vmap(grid_search_run_vmap, in_axes=in_axes)
            grid_search_run_vmap = jax.jit(grid_search_run_vmap)
            if estimator_config["name"] in ["LGPUCB", "GPRegressor"]:
                input_values = (
                    jnp.array(estimator_config["nll_regularization_penalty"])
                    if "nll_regularization_penalty" in estimator_config
                    else jnp.array([0.0]),
                    jnp.array(estimator_config["rkhs_norm_ub"])
                    if "rkhs_norm_ub" in estimator_config
                    else jnp.array([0.0]),
                    jnp.array(estimator_config["kernel_params"]["variance"]),
                    jnp.array(estimator_config["kernel_params"]["length_scale"]),
                    jnp.array([0.0]),
                )
            elif estimator_config["name"] in ["LogisticUCB1"]:
                input_values = (
                    jnp.array([0.0]),
                    jnp.array([0.0]),
                    jnp.array([0.0]),
                    jnp.array([0.0]),
                    jnp.array(estimator_config["arm_norm_ub"]),
                )
            else:
                input_values = (
                    jnp.array([0.0]) for _ in range(5)
                )
            results = jax.block_until_ready(
                grid_search_run_vmap(
                    jax.random.split(rng, config["num_seeds"]),
                    *input_values,
                )
            )
        else:
            run_experiment_vmap = jax.jit(
                jax.vmap(
                    lambda x: run_experiment(
                        x, config, estimator_config, estimator_config["num_iter"]
                    )
                )
            )
            results = jax.block_until_ready(
                run_experiment_vmap(jax.random.split(rng, config["num_seeds"]))
            )
        print(
            "Running time: {:.2f}m {:.2f}s".format(
                *divmod(time.time() - start_time, 60)
            )
        )
        # Save the results
        output_file = os.path.join(args.dir, estimator_config["name"] + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(results[1], f)
