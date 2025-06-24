import jax
import jax.numpy as jnp
from flax import struct

from typing import Callable, Union, Tuple

from src.environments.DuelingEnvironment.DuellingEnv import DuellingEnv, DuellingEnvParams


@struct.dataclass
class UtilityDuellingParams(DuellingEnvParams):
    utility_function_params: Union[jnp.ndarray, float, struct.dataclass] = None
    best_arm_utility: jnp.ndarray = None


@struct.dataclass
class UtilityDuellingEnv(DuellingEnv):
    utility_function: Callable[
        [jnp.ndarray, Union[jnp.ndarray, float, struct.dataclass]], jnp.ndarray
    ] = None
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.sigmoid

    @property
    def default_params(self) -> UtilityDuellingParams:
        return UtilityDuellingParams()

    def _calculate_prob(
        self,
        arm1: jnp.ndarray,
        arm2: jnp.ndarray,
        params: UtilityDuellingParams,
    ) -> jnp.ndarray:
        """
        Calculates the probability of success for the given arm
            :param arm1: First arm to pull, element of domain
            :param arm2: Second arm to pull, element of domain
            :param params: DuellingEnvParams
        """
        arm1 = self.domain.project(arm1)
        arm2 = self.domain.project(arm2)
        utility1 = self.utility_function(arm1, params.utility_function_params)
        utility2 = self.utility_function(arm2, params.utility_function_params)
        return self.activation_function(utility1 - utility2)

    def best_arm(self, params: UtilityDuellingParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the best arm and its utility value in the domain.
        :param params:
        :return:
        """
        def utility_f(arm: jnp.ndarray) -> jnp.ndarray:
            arm = self.domain.project(arm)
            return self.utility_function(arm, params.utility_function_params)

        return self.domain.maximize(utility_f)
