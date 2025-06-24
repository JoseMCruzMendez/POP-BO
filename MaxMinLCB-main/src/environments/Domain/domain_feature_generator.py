import jax.numpy as jnp
import jax
from typing import Dict
from functools import partial


@partial(jax.jit, static_argnames=["norm_ub"])
def scale_arm_norm(arm_features: jax.Array, norm_ub: jax.Array) -> jax.Array:
    """
    Scale the arm feature for the given norm upper bound.
    If norm_ub is None, then no scaling is done.
    :param arm_features: The arm features to scale. Shape: (num_arms, feature_dim)
    :param norm_ub: The norm upper bound.
    :return: The scaled arm features.
    """
    if norm_ub is None:
        return arm_features
    else:
        return (
            norm_ub
            * arm_features
            / jnp.linalg.norm(arm_features, 2, axis=-1, keepdims=True)
        )


"""
DOMAIN FEATURE GENERATORS
"""


@partial(jax.jit, static_argnames=["config"])
def normal(rng: jax.random.PRNGKey, config: Dict) -> jax.Array:
    """
    Generate normal features for the arms.
    :param rng: The random key.
    :param config: The domain config. Must contain the following keys:
        - num_arms: The number of arms.
        - feature_dim: The feature dimension.
        - domain: The domain config. Must contain the following keys:
            - params: The parameters for the normal distribution.
                - mean: The mean of the normal distribution.
                - std: The standard deviation of the normal distribution.
            - norm_ub: The norm upper bound.
    :return: The arm features. Shape: (num_arms, feature_dim)
    """
    config_domain = config["domain"]
    arm_features = jax.random.normal(
        rng, shape=(config["num_arms"], config["feature_dim"])
    )
    arm_features = (
        config_domain["params"]["mean"] + config_domain["params"]["std"] * arm_features
    )
    arm_features = scale_arm_norm(arm_features, config_domain["norm_ub"])
    return arm_features


def uniform(rng: jax.random.PRNGKey, config: Dict) -> jax.Array:
    """
    Generate uniform features for the arms.
    :param rng: The random key.
    :param config: The domain config. Must contain the following keys:
        - num_arms: The number of arms.
        - feature_dim: The feature dimension.
        - domain: The domain config. Must contain the following keys:
            - params: The parameters for the uniform distribution.
                - min: The minimum value of the uniform distribution.
                - max: The maximum value of the uniform distribution.
            - norm_ub: The norm upper bound.
    """
    config_domain = config["domain"]
    arm_features = jax.random.uniform(
        rng, shape=(config["num_arms"], config["feature_dim"])
    )
    # max-min scaling
    arm_features = (
        config_domain["params"]["min"]
        + (config_domain["params"]["max"] - config_domain["params"]["min"])
        * arm_features
    )
    arm_features = scale_arm_norm(arm_features, config_domain["norm_ub"])
    return arm_features


def meshgrid(
    rng: jax.random.PRNGKey,
    domain_range: jax.Array,
    num_arms_per_dim: int,
    num_dims: int,
) -> jax.Array:
    """
    Generate meshgrid (uniform grid) features for the arms.
    :param rng: The random key.
    :param domain_range: The domain range. Shape: (feature_dim, 2).
    :param num_arms_per_dim: The number of arms per dimension.
    :param num_dims: The number of dimensions.
    :return: The arm features. Shape: (num_arms, feature_dim)
    """
    domain_range = jnp.atleast_2d(jnp.array(domain_range))
    assert num_dims == domain_range.shape[0]
    mesh = jnp.meshgrid(
        *[jnp.linspace(*domain_range[i], num_arms_per_dim) for i in range(num_dims)]
    )
    arm_features = jnp.stack([m.ravel() for m in mesh], axis=-1)
    return arm_features
