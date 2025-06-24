import jax
import jax.numpy as jnp
from flax import struct

from typing import Callable, Union, Tuple, Optional

from src.environments.DuelingEnvironment.UtilityDuellingEnvironment import UtilityDuellingEnv, UtilityDuellingParams

def _raise_if_nan(x):
    asin = jnp.arcsin(x/5)
    if jnp.isnan(asin).any():
        raise ValueError("x is out of bounds")

@jax.jit
def error_if_nan(x):
    jax.debug.callback(_raise_if_nan, x)

@struct.dataclass
class GPODuellingEnv(UtilityDuellingEnv):
    utility_function: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    pref_function: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None #Monkey-patch fix for best_arm requiring a 1-parameter function for BT instead of the preference model
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.sigmoid

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
        #For some reason, the base library wraps their utility function in a get_feature call, added here as their wrapper assumes single-input functions
        arm1 = self.domain.get_feature(arm1)
        arm2 = self.domain.get_feature(arm2)
        score = self.pref_function(arm1, arm2)
        #score = self.pref_function(arm2, arm1)
        score = jax.lax.clamp(score, -6., 6.)

        return self.activation_function(score)

    def regret(
            self,
            arm1: jnp.ndarray,
            arm2: jnp.ndarray,
            params: UtilityDuellingParams,
            arm_set: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculates the regret for the given arm.
        If arm_set is not None, then the regret is calculated for the given arm set.
            :param arm1: First arm to compare, element of self.domain
            :param arm2: Second arm to compare, element of self.domain
            :param params: DuellingEnvParams
            :param arm_set: Arm set to compare, Shape: (n, 2), Dtype: jnp.int32
                Describes the n arms to compare
        """

        def compare_arm_to_best(single_arm):
            arm_util = self.utility_function(arm1, params.utility_function_params)
            best_util = self.utility_function(params.best_arm, params.utility_function_params)
            return self.activation_function(arm_util - best_util)
        regrets = jnp.array([compare_arm_to_best(arm1), compare_arm_to_best(arm2)])  # Shape: (2,)
        # best_val = self.utility_function(params.best_arm, params.utility_function_params)
        # arm1_val = self.utility_function(arm1, params.utility_function_params)
        # arm2_val = self.utility_function(arm2, params.utility_function_params)
        # regrets = jnp.array([best_val - arm1_val, best_val - arm2_val])

        if arm_set is None:
            return regrets
        else:
            raise NotImplementedError