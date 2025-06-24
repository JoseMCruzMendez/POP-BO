from src.environments.LogisticEnvironment.BanditEnvironment import (
    BanditEnvironment,
    EnvParams,
)
# from src.environments.utils import get_utility
from flax import struct
import jax.numpy as jnp

import jax
from typing import Union, Callable, Optional, Tuple


@struct.dataclass
class LogisticEnvParams(EnvParams):
    utility_function_params: Union[jnp.ndarray, float, struct.dataclass] = None


@struct.dataclass
class UtilityLogisticBanditEnvironment(BanditEnvironment):
    """Logistic Bandit Environment."""
    utility_function: Callable[
        [jnp.ndarray, Union[jnp.ndarray, float, struct.dataclass]], jnp.ndarray
    ] = None
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.sigmoid

    @property
    def default_params(self) -> struct.dataclass:
        return LogisticEnvParams()

    def _calculate_prob(
        self, arm: jnp.ndarray, params: LogisticEnvParams
    ) -> jnp.ndarray:
        """
        Calculates the probability of success for the given arm.
        :param arm: The arm index to calculate the probability for.
        :param params: The environment parameters.
        :return: The probability of success for the given arm.
        """
        arm = self.domain.project(arm)
        utility = self.utility_function(
            arm, params.utility_function_params
        )
        return self.activation_function(utility)

    def pull(
        self, key: jax.random.PRNGKey, arm: Union[int, jnp.ndarray], params: LogisticEnvParams
    ) -> jnp.ndarray:
        """Performs an arm in the environment. Returns the reward."""
        key, key_sample = jax.random.split(key)
        prob = self._calculate_prob(arm, params)
        reward = jax.random.bernoulli(key_sample, prob)
        return reward

    def regret(
        self,
        arm: Union[int, jnp.ndarray],
        params: LogisticEnvParams,
        arm_set: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculates the regret for the given arm."""
        prob = self._calculate_prob(arm, params)
        if arm_set is None:
            prob_best = params.best_arm_value
        else:
            probs = jax.vmap(self._calculate_prob, in_axes=(0, None))(arm_set, params)
            prob_best = jnp.max(probs)
        return prob_best - prob

    def best_arm(
        self,
        params: LogisticEnvParams,
    ) -> Tuple[int, jnp.ndarray]:
        """Returns the best arm."""
        return self.domain.maximize(lambda x: self._calculate_prob(x, params))


