import jax.numpy as jnp
import time
from typing import Tuple, Callable, Optional
import pickle
import os
import yaml
import argparse
from chex import PRNGKey


import jax

from run_experiment_logistic import UTILITY_FUNCTIONS
from src.utils.experiment import initialize_estimator
from src.utils.utility_functions import load_yelp_data, create_yelp_utility

from src.environments.Domain.DiscreteDomain import DiscreteDomain
from src.environments.Domain import domain_feature_generator
from src.environments.DuelingEnvironment.UtilityDuellingEnvironment import (
    UtilityDuellingEnv,
    DuellingEnvParams,
)

from src.bandits.EmpiricalMean import EmpiricalMean
from src.bandits.LGPUCB import LGPUCB

from src.duellingBandits.acquisition_functions import (
    max_min_lcb,
    min_max_ucb,
    max_max_ucb,
    max_info_gain,
    MultiSBM,
    information_directed_sampling,
    Doubler,
    RUCB,
    Doubler_single_estimator,
    Sparring,
    doubleTS
)


ESTIMATORS = {
    "EmpiricalMean": EmpiricalMean,
    "LGPUCB": LGPUCB,
}

ACQUISITION_FUNCTIONS = {
    "max_min_lcb": max_min_lcb,
    "max_min_lcb_no_candidates": lambda *args, **kwargs: max_min_lcb(
        *args, **kwargs, use_candidate_set=False
    ),
    "min_max_ucb": min_max_ucb,
    "max_max_ucb": max_max_ucb,
    "max_info_gain": max_info_gain,
    "MultiSBM": MultiSBM,
    "IDS": information_directed_sampling,
    "Doubler": Doubler,
    "Doubler_no_candidates": lambda *args, **kwargs: Doubler(
        *args, **kwargs, use_candidate_set=False
    ),
    "Doubler_single_estimator": Doubler_single_estimator,
    "RUCB": RUCB,
    "Sparring": Sparring,
    "doubleTS": doubleTS,
}



def initialize_environment(
    rng: PRNGKey, config: dict, env_data: Optional[dict] = None
) -> Tuple[DiscreteDomain, UtilityDuellingEnv, DuellingEnvParams]:
    # Setup domain and environment
    if config["utility_function"] != "yelp":
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
            raise ValueError("Invalid arm_initialization. Use 'normal' or 'uniform'.")
        domain = DiscreteDomain.create(
            num_elements=config["num_arms"], features=arm_features
        )
        rng, _rng = jax.random.split(rng)
        utility_params, utility_function = UTILITY_FUNCTIONS[
            config["utility_function"]
        ](
            _rng,
            domain,
            config["utility_function_params"],
        )
    elif config["utility_function"] == "yelp":
        rng, _rng = jax.random.split(rng)
        arm_features, utilities = env_data["arm_features"], env_data["utilities"]
        domain = DiscreteDomain.create(
            num_elements=arm_features.shape[0], features=arm_features
        )
        utility_params, utility_function = create_yelp_utility(
            _rng,
            domain,
            config["utility_function_params"],
            utilities,
        )
    else:
        raise ValueError("Invalid param_initialization. Use 'normal' or 'yelp'.")

    env = UtilityDuellingEnv(
        domain=domain,
        utility_function=utility_function,
        # use_domain_features=(config["utility_function"] != "yelp"),
    )
    env_params = env.default_params.replace(
        utility_function_params=utility_params,
    )
    best_arm, best_arm_utility = env.best_arm(env_params)
    env_params = env_params.replace(
        best_arm=best_arm, best_arm_utility=best_arm_utility
    )

    # print("Best arm: ", best_arm, "Best arm utility: ", best_arm_utility)
    return domain, env, env_params


def make_experiment_runner(
    config: dict,
    acquisition_function_name: str,
    acquisition_function: Callable = None,
    acquisition_function_idx: int = None,
    return_env_and_estimator: bool = False,
):
    assert acquisition_function is not None or acquisition_function_idx is not None

    def run_experiment(
        rng: PRNGKey,
        env_data: Optional[dict] = None,
        estimator_config_update: Optional[dict] = None,
    ):
        # Initialize environment
        rng, _rng = jax.random.split(rng)
        discrete_domain, env, env_params = initialize_environment(
            rng, config, env_data=env_data
        )

        # Initialize estimator
        rng, _rng = jax.random.split(rng)
        config_estimator = config["estimator"]
        if estimator_config_update is not None:
            config_estimator.update(estimator_config_update)
        if acquisition_function_name in ["Doubler_single_estimator"]:  # Single Estimator
            estimator, estimator_params = initialize_estimator(
                _rng, config, config_estimator, env.domain, duelling=False
            )
        elif acquisition_function_name == "Sparring":  # Two Independent Estimators
            _rng1, _rng2 = jax.random.split(_rng)
            estimator1, estimator_params1 = initialize_estimator(
                _rng1, config, config_estimator, env.domain, duelling=False
            )
            estimator2, estimator_params2 = initialize_estimator(
                _rng2, config, config_estimator, env.domain, duelling=False
            )
            estimator = (estimator1, estimator2)
            estimator_params = (estimator_params1, estimator_params2)
        else:  # Duelling Estimator
            estimator, estimator_params = initialize_estimator(
                _rng, config, config_estimator, env.domain, duelling=True
            )

        # Create function to be used with jax.lax.scan for the loop
        def loop_body(carry, _):
            rng, estimator_params, acquisition_info = carry
            if acquisition_function_name == "Sparring":
                posterior_mean1, posterior_var1, beta1 = estimator[0].estimate(
                    estimator_params[0]
                )
                posterior_mean2, posterior_var2, beta2 = estimator[1].estimate(
                    estimator_params[1]
                )
                posterior_mean = (posterior_mean1, posterior_mean2)
                posterior_var = (posterior_var1, posterior_var2)
                beta = (beta1, beta2)
            else:
                posterior_mean, posterior_var, beta = estimator.estimate(
                    estimator_params
                )
            rng, _rng = jax.random.split(rng)
            if acquisition_function is not None:
                arm1, arm2, acquisition_info = acquisition_function(
                    _rng,
                    posterior_mean,  # Shape: (n_arms, n_arms)
                    posterior_var,
                    beta=beta,
                    acquisition_info=acquisition_info,
                    rho2=estimator.kappa * estimator_params.lambda_
                    if acquisition_function_name != "Sparring"
                    else None,
                )
            else:
                arm1, arm2, acquisition_info = jax.lax.switch(
                    acquisition_function_idx,
                    ACQUISITION_FUNCTIONS.values(),
                    _rng,
                    posterior_mean,
                    posterior_var,
                    beta,
                    acquisition_info=acquisition_info,
                )

            rng, _rng = jax.random.split(rng)
            reward = env.pull(_rng, arm1, arm2, env_params)
            regrets = env.regret(arm1, arm2, env_params)  # Shape: (2,)

            # Update estimator symmetrically, if arm1 != arm2
            def update_estimators(_rng, arms, reward, params):
                if acquisition_function_name in ["Doubler_single_estimator"]:
                    _rng1, _rng2 = jax.random.split(_rng)
                    params_update, _ = estimator.update(_rng1, arms[0], reward, params)
                    params_update, update_info = estimator.update(
                        _rng2, arms[1], 1 - reward, params_update
                    )
                elif acquisition_function_name == "Sparring":
                    _rng1, _rng2 = jax.random.split(_rng)
                    params_update1, _ = estimator[0].update(
                        _rng1, arms[0], reward, params[0]
                    )
                    params_update2, update_info = estimator[1].update(
                        _rng2, arms[1], 1 - reward, params[1]
                    )
                    params_update = (params_update1, params_update2)
                else:
                    params_update, update_info = estimator.update(
                        _rng, arms, reward, params
                    )
                return params_update, update_info

            rng, _rng = jax.random.split(rng)
            estimator_params, update_info = jax.lax.cond(
                arm1 == arm2,
                lambda x: (
                    x[-1],
                    {
                        "alpha": jnp.full_like(estimator_params.alpha, jnp.nan)
                        if acquisition_function_name != "Sparring"
                        else jnp.full_like(estimator_params[0].alpha, jnp.nan),
                        "nll": jnp.nan,
                    }
                    if config["estimator"]["name"] == "LGPUCB"
                    else {},
                ),
                lambda x: update_estimators(*x),
                (_rng, (arm1, arm2), reward, estimator_params),
            )

            return (rng, estimator_params, acquisition_info), {
                "selected_arm": (arm1, arm2),
                "reward": reward,
                "regrets": regrets,
            }

        if acquisition_function_name in ["MultiSBM"]:
            acquisition_info = {"previous_arm": -1}
        elif acquisition_function_name in ["Doubler_single_estimator"]:
            rng, _rng = jax.random.split(rng)
            random_idx = jax.random.choice(_rng, discrete_domain.num_elements, shape=(1,))
            acquisition_info = {
                "i": 1,
                "j": 1,
                "selection_set": jnp.zeros(discrete_domain.num_elements).at[random_idx].set(True),
                "next_set": jnp.zeros(discrete_domain.num_elements),
            }
        else:
            acquisition_info = {}

        carry, outputs = jax.lax.scan(
            loop_body,
            (rng, estimator_params, acquisition_info),
            None,
            length=config["num_iter"],
        )
        (rng, estimator_params, _) = carry
        if return_env_and_estimator:
            return {
                "env": (env, env_params),
                "estimator": (estimator, estimator_params),
            }, outputs
        else:
            return {"env": env_params, "estimator": estimator_params}, outputs

    return run_experiment


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run duelling bandit experiment.")
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

    acquisition_function_list = config["acquisition_functions"] if args.algo is None else [args.algo]
    print("Acquisition functions: ", acquisition_function_list)

    rng = jax.random.PRNGKey(config["seed"])

    # If using Yelp data, load it
    if config["utility_function"] == "yelp":
        print("Loading Yelp data")
        config_utility_params = config["utility_function_params"]
        env_data = load_yelp_data(
            data_dir="data/yelp_aggregates",
            city=config_utility_params["city"],
            min_review_count=config_utility_params["min_review_count"],
            min_review_per_user=config_utility_params["min_review_per_user"],
            collaborative_filtering_args=config_utility_params[
                "collaborative_filtering_args"
            ],
        )
        print("Data loaded. Shapes:")
        for key, data in env_data.items():
            print(key, data.shape)
    else:
        env_data = None

    for acquisition_function_name in acquisition_function_list:
        try:
            acquisition_function = ACQUISITION_FUNCTIONS[acquisition_function_name]
        except KeyError:
            raise ValueError(
                f"Invalid acquisition function name: {acquisition_function_name}"
            )
        print(
            f"Running experiment for acquisition function: {acquisition_function_name}"
        )
        start_time = time.time()
        run_experiment = make_experiment_runner(
            config=config,
            acquisition_function_name=acquisition_function_name,
            acquisition_function=acquisition_function,
        )

        # Get estimator params to update in a grid search
        config_estimator = config["estimator"].copy()
        config_estimator_update = {
            name: jnp.array(value)
            for name, value in config_estimator.items()
            if isinstance(value, list)
        }
        if any(
            [
                isinstance(value, list)
                for value in config_estimator["kernel_params"].values()
            ]
        ):
            config_estimator_update["kernel_params"] = {
                name: jnp.array(value)
                for name, value in config_estimator["kernel_params"].items()
                if isinstance(value, list)
            }
        flat_tree, tree_structure = jax.tree_util.tree_flatten(config_estimator_update)
        grid_search_mesh = jax.tree_map(lambda x: x.ravel(), jnp.meshgrid(*flat_tree))
        grid_search_params = jax.tree_util.tree_unflatten(
            tree_structure, grid_search_mesh
        )
        # print("Grid search params: ", grid_search_params)

        if len(grid_search_params) > 0:
            print("Grid search params: ", grid_search_params)
            run_experiment_vmap = jax.jit(
                jax.vmap(
                    jax.vmap(
                        run_experiment,
                        in_axes=(None, None, 0),
                    ),
                    in_axes=(0, None, None),
                )
            )
            output, metrics = jax.block_until_ready(
                run_experiment_vmap(
                    jax.random.split(rng, config["num_seeds"]),
                    env_data,
                    grid_search_params,
                )
            )
        else:
            run_experiment_vmap = jax.vmap(run_experiment, in_axes=(0, None, None))
            output, metrics = jax.block_until_ready(
                run_experiment_vmap(
                    jax.random.split(rng, config["num_seeds"]),
                    env_data,
                    None,
                )
            )
        print(
            "Running time: {:.2f}m {:.2f}s".format(
                *divmod(time.time() - start_time, 60)
            )
        )

        # Save the results
        output_file = os.path.join(args.dir, acquisition_function_name + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(metrics, f)

        # Save the estimator params
        output_file = os.path.join(
            args.dir, acquisition_function_name + "_estimator_params.pkl"
        )
        with open(output_file, "wb") as f:
            pickle.dump(output["estimator"], f)

        if len(grid_search_params) > 0:
            # Save grid_search_params
            output_file = os.path.join(
                args.dir, acquisition_function_name + "_grid_search_params.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(grid_search_params, f)

        del output, metrics, grid_search_params
