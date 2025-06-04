from src.bandits.BaseEstimator import BaseEstimator, EstimatorParams
from src.environments.Domain.DiscreteDomain import DiscreteDomain
from typing import Union, Tuple
import jax.numpy as jnp
import jax
from chex import PRNGKey

from flax import struct


@struct.dataclass
class EmpiricalMeanParams(EstimatorParams):
    delta: float = None
    beta: float = 1.0
    state_visits: jnp.ndarray = None
    mean_estimates: jnp.ndarray = None


@struct.dataclass
class EmpiricalMean(BaseEstimator):
    """Simple UCB Algorithm (Algorithm 3) from the Bandit Algorithms by Tor Lattimore and Csaba Szepesvari."""
    initial_state_visits: jnp.ndarray
    initial_mean_estimates: jnp.ndarray
    duelling: bool

    @classmethod
    def create(
        cls,
        domain: DiscreteDomain,
        mean_initialization_values: Union[float, jnp.ndarray] = 0.5,
        duelling: bool = False,
    ):
        """Creates an EmpiricalMean estimator with the given domain and delta."""
        assert isinstance(domain, DiscreteDomain)
        n_arms = domain.num_elements
        if duelling:
            shape = (n_arms, n_arms)
        else:
            shape = (n_arms,)
        if isinstance(mean_initialization_values, float):
            mean_initialization_values = mean_initialization_values * jnp.ones(shape)
        assert mean_initialization_values.shape == shape
        return cls(
            domain=domain,
            initial_state_visits=jnp.zeros(shape),
            initial_mean_estimates=mean_initialization_values,
            duelling=duelling,
        )

    @property
    def default_params(self):
        return EmpiricalMeanParams()

    def reset(self, rng: PRNGKey, params: EmpiricalMeanParams):
        """Resets the estimator to its initial state."""
        return self.default_params.replace(
            state_visits=self.initial_state_visits,
            mean_estimates=self.initial_mean_estimates,
        )

    def _get_arm_mask(self, arm: int) -> jnp.ndarray:
        return (arm == jnp.arange(self.domain.num_elements))

    def update(
        self,
        key: PRNGKey,
        arm: Union[int, Tuple[int, int]],
        feedback: int,
        params: EmpiricalMeanParams,
    ):
        """Updates the state_visits and mean_estimates for the given arm and feedback."""
        state_visits, mean_estimates = params.state_visits, params.mean_estimates
        arm_mask = self._get_arm_mask(arm)
        mean_estimates = jnp.where(
            arm_mask,
            (mean_estimates * state_visits + feedback) / (state_visits + 1),
            mean_estimates,
        )
        state_visits = state_visits + arm_mask
        return (
            params.replace(state_visits=state_visits, mean_estimates=mean_estimates),
            None,
        )

    @staticmethod
    def get_bonus(params: EmpiricalMeanParams) -> jnp.ndarray:
        """Returns the estimated mean and bonus for the given arm."""
        return jnp.sqrt(
            2
            * jnp.log(1 / params.delta)
            / jnp.clip(params.state_visits, a_min=0.0001, a_max=None)
        )

    def estimate(
        self, params: EmpiricalMeanParams
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        return params.mean_estimates, self.get_bonus(params), params.beta

    def best_arm(
        self, key: PRNGKey, params: EmpiricalMeanParams
    ) -> Tuple[jnp.ndarray, jnp.ndarray, EmpiricalMeanParams]:
        """Returns the best arm."""
        mean, bonus, beta = self.estimate(params)
        best_arm, best_ucb_value = self.domain.maximize(
            lambda idx: mean[idx] + beta * bonus[idx]
        )
        return best_arm, mean, bonus, params


@struct.dataclass
class EmpiricalMeanDuelling(EmpiricalMean):
    """EmpiricalMeanDuelling is the EmpiricalMean estimator for the duelling bandit setting."""
    def _get_arm_mask(self, arm: Tuple[int, int]) -> jnp.ndarray:
        return (
                (arm[0] == jnp.arange(self.domain.num_elements))[:, None]
                & (arm[1] == jnp.arange(self.domain.num_elements))[None, :]
            )
