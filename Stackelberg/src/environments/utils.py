import jax
import jax.numpy as jnp
from flax import struct

from src.environments.Domain.DiscreteDomain import DiscreteDomain
from typing import Union, Callable


def get_features_from_domain(arm: jnp.ndarray, domain: DiscreteDomain) -> jnp.ndarray:
    """
    Projects the given arm to the domain and returns its feature.
    """
    arm = domain.project(arm)  # map to domain
    if isinstance(domain, DiscreteDomain):
        if domain.has_features:
            arm_feature = domain.get_feature(arm)
        else:
            arm_feature = jax.nn.one_hot(arm, domain.num_elements)
    else:
        raise NotImplementedError
    return arm_feature


def get_utility(
        arm: jnp.ndarray,
        domain: DiscreteDomain,
        utility_function: Callable[[jnp.ndarray, Union[jnp.ndarray, float, struct.dataclass]], jnp.ndarray],
        utility_function_params: Union[jnp.ndarray, float, struct.dataclass],
        use_domain_features: bool = True,
) -> jnp.ndarray:
    """
    Calculates the features and then the utility of the given arm.
    """
    if use_domain_features:
        arm_feature = get_features_from_domain(arm, domain)
    else:
        arm_feature = arm
    return utility_function(
        arm_feature, utility_function_params
    )
