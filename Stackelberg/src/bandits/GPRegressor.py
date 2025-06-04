from flax import struct
import jax.numpy as jnp
import jax
from chex import PRNGKey

from src.utils.kernels import Kernel

from typing import Tuple, Dict, Union

from src.bandits.BaseEstimator import BaseEstimator
from src.environments.Domain.DiscreteDomain import DiscreteDomain


@struct.dataclass
class GPRegressorParams:
    # Hyperparameters
    kernel: Kernel = None
    lambda_: float = None
    beta: float = jnp.nan
    delta: float = None
    # Variables
    gram_matrix: jnp.ndarray = None
    # gram_matrix_L: jnp.ndarray = None  # TODO: Finish implementation for speed efficiency
    # Counters and memory
    ctr: int = 0
    arms: jnp.ndarray = None
    rewards: jnp.ndarray = None


@struct.dataclass
class GPRegressor(BaseEstimator):
    horizon: int
    duelling: bool
    algo_name: str = "GPRegressor"

    @classmethod
    def create(
        cls,
        domain: DiscreteDomain,
        horizon: int,
        duelling: bool = False,
    ):
        return cls(
            domain=domain,
            horizon=horizon,
            duelling=duelling,
        )

    @property
    def default_params(self):
        return GPRegressorParams()

    def reset(self, rng: PRNGKey, params: GPRegressorParams):
        params = params.replace(
            gram_matrix=jnp.full((self.horizon, self.horizon), jnp.nan),
            ctr=0,
            arms=jnp.full((self.horizon, 2), jnp.nan)
            if self.duelling
            else jnp.full(self.horizon, jnp.nan),
            rewards=jnp.full(self.horizon, jnp.nan),
        )
        return params

    def _update_params(
        self, arm: Union[int, Tuple[int, int]], feedback: int, params: GPRegressorParams
    ) -> GPRegressorParams:
        """Update the arms, rewards, and gram_matrix in params."""
        # Update arms, rewards, and gram_matrix
        rewards_updated = params.rewards.at[params.ctr].set(feedback)
        arms_updated = params.arms.at[params.ctr].set(arm)

        if self.duelling:
            history_feature_matrix = (
                self.domain.get_feature(arms_updated[:, 0]),
                self.domain.get_feature(arms_updated[:, 1]),
            )  # Tuple of 2, Shape: (horizon, dim), with NaNs
            arm_feature = (
                self.domain.get_feature(arm[0]),
                self.domain.get_feature(arm[1]),
            )  # Tuple of 2, Shape: (dim,)
        else:
            history_feature_matrix = self._get_arm_feature(arms_updated)
            arm_feature = self._get_arm_feature(arm)
        # history_feature_matrix = self.domain.get_feature(arms_updated)
        # arm_feature = self.domain.get_feature(arm)
        cross_variance = params.kernel.cross_covariance(
            history_feature_matrix, arm_feature
        )[
            :, 0
        ]  # Shape: (horizon,)
        gram_matrix_updated = params.gram_matrix.at[params.ctr, :].set(cross_variance)
        gram_matrix_updated = gram_matrix_updated.at[:, params.ctr].set(cross_variance)

        # TODO: Add a new row to the Cholesky decomposition in gram_matrix_L following the Cholesky-Banachiewicz algorithm
        # https: // en.wikipedia.org / wiki / Cholesky_decomposition

        params = params.replace(
            arms=arms_updated,
            rewards=rewards_updated,
            gram_matrix=gram_matrix_updated,
            ctr=params.ctr + 1,
        )
        return params

    def _get_arm_feature(self, arm: Union[int, jnp.ndarray]) -> jnp.ndarray:
        return self.domain.get_feature(arm)

    def update(
        self,
        key: PRNGKey,
        arm: Union[int, Tuple[int, int]],
        feedback: int,
        params: GPRegressorParams,
    ) -> Tuple[GPRegressorParams, Dict[str, jnp.ndarray]]:
        params = self._update_params(arm, feedback, params)
        return params, {}

    def _get_cross_variance(
        self,
        history_feature_matrix: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        params: GPRegressorParams,
    ):
        """Get the cross variance between the history and feature_matrix."""
        if self.duelling:
            duelling_cross_variance_vmap = jax.vmap(
                jax.vmap(
                    lambda x, y: params.kernel.cross_covariance(
                        history_feature_matrix, (x, y)
                    )[
                        :, 0
                    ],  # Shape: (horizon, )
                    in_axes=(None, 0),
                    out_axes=1,
                ),  # Shape: (horizon, n_arms (y))
                in_axes=(0, None),
                out_axes=1,
            )  # Shape: (horizon, n_arms (x), n_arms (y))

            cross_variance = duelling_cross_variance_vmap(
                feature_matrix, feature_matrix
            )  # Shape: (horizon, n_arms, n_arms)
        else:
            cross_variance = params.kernel.cross_covariance(
                history_feature_matrix, feature_matrix
            )  # Shape: (horizon, n_arms)
        return cross_variance

    def _get_gram_matrix(self, feature_matrix: jnp.ndarray, params: GPRegressorParams):
        """Get the gram matrix."""
        if self.duelling:
            duelling_self_gram_vmap = jax.vmap(
                jax.vmap(
                    lambda x, y: params.kernel.gram((x, y))[0, 0],  # Shape: (, )
                    in_axes=(None, 0),
                ),  # Shape: (n_arms (y), )
                in_axes=(0, None),
            )  # Shape: (n_arms (x), n_arms (y))
            return duelling_self_gram_vmap(
                feature_matrix, feature_matrix
            )  # Shape: (n_arms, n_arms)
        else:
            return jax.vmap(params.kernel.gram)(feature_matrix)[:, 0, 0]

    @staticmethod
    def _gp_posterior_variance(
        cholesky_lower_diagonal: jnp.ndarray,
        cross_variance: jnp.ndarray,
        self_k: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the GP
        :param cholesky_lower_diagonal: Shape: (horizon, horizon)
        :param cross_variance: Shape: (horizon,)
        :param self_k: Shape: ()
        :return: mean: Shape: ()
        """
        return self_k - jnp.sum(
            jnp.square(
                jax.scipy.linalg.solve_triangular(
                    cholesky_lower_diagonal, cross_variance, lower=True
                )
            ),
            axis=0,
        )

    @staticmethod
    def _gp_posterior_mean(
        cholesky_lower_diagonal: jnp.ndarray,
        cross_variance: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the GP
        :param cholesky_lower_diagonal: Shape: (horizon, horizon)
        :param cross_variance: Shape: (horizon,)
        :param rewards: Shape: (horizon,)
        :return: mean: Shape: ()
        """
        cholesky_lower_diagonal = cholesky_lower_diagonal + jnp.diag(
            jnp.where(jnp.isnan(rewards), 1.0, 0.0)
        )  # Add diagonal to cholesky_lower_diagonal for numerical stability
        m = jax.scipy.linalg.solve_triangular(
            cholesky_lower_diagonal, rewards, lower=True
        )
        alpha = jax.scipy.linalg.solve_triangular(
            cholesky_lower_diagonal.T, jnp.nan_to_num(m, nan=0.0), lower=False
        )
        return jnp.dot(cross_variance, alpha)

    def _posterior_mean(
        self,
        cholesky_lower_diagonal: jnp.ndarray,
        cross_variance: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """Posterior mean of the GP."""
        if self.duelling:
            return jax.vmap(
                jax.vmap(self._gp_posterior_mean, in_axes=(None, 1, None)),
                in_axes=(None, 1, None),
            )(
                cholesky_lower_diagonal,
                jnp.nan_to_num(cross_variance, nan=0.0),
                rewards,
            )  # Shape: (n_arms, n_arms)
        else:
            return jax.vmap(self._gp_posterior_mean, in_axes=(None, 1, None))(
                cholesky_lower_diagonal,
                jnp.nan_to_num(cross_variance, nan=0.0),
                rewards,
            )  # Shape: (n_arms,)

    def _posterior_var(
        self,
        cholesky_lower_diagonal: jnp.ndarray,
        cross_variance: jnp.ndarray,
        self_gram: jnp.ndarray,
    ) -> jnp.ndarray:
        """Posterior variance of the GP."""
        if self.duelling:
            return jax.vmap(
                jax.vmap(self._gp_posterior_variance, in_axes=(None, 1, 0)),
                in_axes=(None, 1, 0),
            )(
                cholesky_lower_diagonal,
                jnp.nan_to_num(cross_variance, nan=0.0),
                self_gram,
            )  # Shape: (n_arms, n_arms)
        else:
            return jax.vmap(self._gp_posterior_variance, in_axes=(None, 1, 0))(
                cholesky_lower_diagonal,
                jnp.nan_to_num(cross_variance, nan=0.0),
                self_gram,
            )  # Shape: (n_arms,)

    def _calculate_beta(
        self, params: GPRegressorParams, cholesky_lower_diagonal: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculate the beta for the UCB. Return nan if beta is nan."""
        return jax.lax.select(
            jnp.isnan(params.beta),
            jnp.nan,
            params.beta,
        )

    def estimate(
        self,
        params: GPRegressorParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Estimate the mean and variance of the posterior distribution."""
        if self.duelling:
            history_feature_matrix = (
                self.domain.get_feature(params.arms[:, 0]),
                self.domain.get_feature(params.arms[:, 1]),
            )  # Tuple of 2, Shape: (horizon, dim), with NaNs
        else:
            history_feature_matrix = self._get_arm_feature(
                params.arms
            )  # Shape: (horizon, dim), with NaNs
        feature_matrix = self.domain.feature_matrix  # Shape: (n_arms, dim)

        # Calculate the covariance matrix
        K = params.gram_matrix  # Shape: (horizon, horizon)
        K_filled_0 = jnp.where(jnp.isnan(K), 0, K)
        covariance_matrix = K_filled_0 + params.lambda_ * jnp.eye(self.horizon)
        L = jnp.linalg.cholesky(covariance_matrix)  # Shape: (horizon, horizon)
        L = L - jnp.where(
            jnp.isnan(params.rewards), jnp.sqrt(params.lambda_), 0.0
        )  # Set diagonals beyond ctr to zero

        # Calculate the cross variance and self_gram
        cross_variance = self._get_cross_variance(
            history_feature_matrix, feature_matrix, params
        )  # Shape: (horizon, n_arms, n_arms) if duelling else (horizon, n_arms)
        self_gram = self._get_gram_matrix(
            feature_matrix, params
        )  # Shape: (n_arms, n_arms) if duelling else (n_arms,)

        # Calculate the posterior mean and variance
        posterior_mean = self._posterior_mean(L, cross_variance, params.rewards)
        posterior_var = self._posterior_var(L, cross_variance, self_gram)

        beta = self._calculate_beta(params, L)
        return posterior_mean, posterior_var, beta

    def best_arm(
        self, key: PRNGKey, params: GPRegressorParams
    ) -> Tuple[int, float, GPRegressorParams]:
        """Return the best arm and its index."""

        def get_best_arm_set():
            posterior_mean, posterior_variance, beta = self.estimate(
                params
            )  # Shape: (num_arms,)
            ucb = posterior_mean + beta * jnp.sqrt(posterior_variance)
            ucb_max = jnp.max(ucb)
            argmax_set = jnp.where(
                jnp.abs(ucb - ucb_max) < 1e-16,
                jax.random.choice(
                    key, ucb.shape[0], shape=(ucb.shape[0],), replace=False
                ),
                -1.0,
            )
            return argmax_set, posterior_mean, posterior_variance

        n = self.domain.num_elements
        argmax_set, posterior_mean, posterior_variance = jax.lax.cond(
            params.ctr == 0,
            lambda: (
                jax.random.choice(
                    key, jnp.arange(n, dtype=jnp.float32), shape=(n,), replace=False
                ),
                0.5 * jnp.ones(n, dtype=jnp.float32),
                jnp.nan * jnp.ones(n, dtype=jnp.float32),
            ),
            get_best_arm_set,
        )
        best_arm = jnp.argmax(argmax_set)
        return best_arm, posterior_mean, posterior_variance, params
