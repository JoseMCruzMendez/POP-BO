import jax
import jax.numpy as jnp
from flax import struct

from typing import Callable, Union, Tuple

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
        error_if_nan(arm1)
        error_if_nan(arm2)
        score = self.pref_function(arm1, arm2)
        score = jax.lax.clamp(score, -3., 3.)

        return self.activation_function(score)