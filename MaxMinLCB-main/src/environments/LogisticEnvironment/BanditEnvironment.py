import jax
import jax.numpy as jnp
from flax import struct

from typing import Union, Optional

from src.environments.Domain.BaseDomain import BaseDomain


@struct.dataclass
class EnvParams:
    best_arm: Union[int, jnp.ndarray] = None
    best_arm_value: Union[float, jnp.ndarray] = None


@struct.dataclass
class BanditEnvironment:
    """Jittable abstract base class for all Bandit Environments."""
    domain: BaseDomain

    @property
    def default_params(self) -> struct.dataclass:
        return EnvParams()

    def pull(
        self,
        key: jax.random.PRNGKey,
        arm: Union[int, jnp.ndarray],
        params: Optional[struct.dataclass],
    ) -> float:
        """Performs an arm in the environment. Returns the reward."""
        raise NotImplementedError

    def regret(
        self,
        arm: Union[int, jnp.ndarray],
        params: struct.dataclass,
        arm_set: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Calculates the regret for the given arm.
        If arm_set is not None, then the regret is calculated for the given arm set.
        """
        raise NotImplementedError

    def best_arm(self, params: EnvParams) -> Union[int, jnp.ndarray]:
        """Returns the best arm."""
        raise NotImplementedError
