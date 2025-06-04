from typing import Union, Tuple, Dict

from flax import struct

import jax
from chex import PRNGKey
import jax.numpy as jnp
from src.environments.Domain.ContinuousDomain import ContinuousDomain
from src.environments.Domain.DiscreteDomain import DiscreteDomain


@struct.dataclass
class EstimatorParams:
    pass


@struct.dataclass
class BaseEstimator:
    domain: Union[DiscreteDomain, ContinuousDomain]

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @property
    def default_params(self):
        return EstimatorParams()

    def reset(self, rng: PRNGKey, params: EstimatorParams):
        raise NotImplementedError

    def update(
        self, key: PRNGKey, arm: int, feedback: int, params: EstimatorParams
    ) -> Tuple[EstimatorParams, Dict[str, jnp.ndarray]]:
        raise NotImplementedError

    def estimate(self, params: EstimatorParams) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        :param params: EstimatorParams
        :return: posterior_mean, posterior_variance, beta
        """
        raise NotImplementedError

    def best_arm(self, key: PRNGKey, params: EstimatorParams):
        raise NotImplementedError
