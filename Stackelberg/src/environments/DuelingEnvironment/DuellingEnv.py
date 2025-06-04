from typing import Tuple, Union, Optional

from flax import struct
from chex import PRNGKey
import jax.numpy as jnp
import jax


@struct.dataclass
class DuellingEnvParams:
    best_arm: jnp.ndarray = None


@struct.dataclass
class DuellingEnv:
    domain: struct.dataclass

    @property
    def default_params(self) -> DuellingEnvParams:
        return DuellingEnvParams()

    @staticmethod
    def _calculate_prob(
        arm1: jnp.ndarray, arm2: jnp.ndarray, params: DuellingEnvParams
    ) -> jnp.ndarray:
        """
        Calculates the probability of success for the given arm.
            # :param arm: Arm to pull, Shape: (2,), Dtype: jnp.int32
            #     Describes the two arms to compare
            :param arm1: First arm to pull, element of self.domain
            :param arm2: Second arm to pull, element of self.domain
            :param params: DuellingEnvParams
        """
        # return params.probabilities[arm[0], arm[1]]
        raise NotImplementedError

    def pull(
        self,
        key: PRNGKey,
        arm1: jnp.ndarray,
        arm2: jnp.ndarray,
        params: DuellingEnvParams,
    ) -> jnp.ndarray:
        """
        Performs an arm in the environment. Returns the reward.
            :param key: PRNGKey
            :param arm1: First arm to pull, element of self.domain
            :param arm2: Second arm to pull, element of self.domain
            :param params: DuellingEnvParams
        """
        probs = self._calculate_prob(arm1, arm2, params)
        return jax.random.bernoulli(key, probs)

    def regret(
        self,
        arm1: jnp.ndarray,
        arm2: jnp.ndarray,
        params: DuellingEnvParams,
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
            return self._calculate_prob(params.best_arm, single_arm, params) - 0.5
        regrets = jnp.array([compare_arm_to_best(arm1), compare_arm_to_best(arm2)])  # Shape: (2,)

        if arm_set is None:
            return regrets
        else:
            raise NotImplementedError

    def best_arm(self, params: DuellingEnvParams) -> jnp.ndarray:
        """Returns the best arm."""
        raise NotImplementedError
