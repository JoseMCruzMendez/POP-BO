from flax import struct
import jax.numpy as jnp
import jax
from chex import PRNGKey
import jaxopt

from src.utils.utils import dsigmoid
from src.utils.kernels import Kernel

from typing import Tuple, Dict, Union

from src.bandits.BaseEstimator import BaseEstimator
from src.environments.Domain.DiscreteDomain import DiscreteDomain


@struct.dataclass
class LGPUCBParams:
    # Hyperparameters
    kernel: Kernel = None
    lambda_: float = None
    beta: float = jnp.nan
    delta: float = None
    nll_regularization_penalty: float = 0.0
    # Variables
    alpha: jnp.ndarray = None
    gram_matrix: jnp.ndarray = None
    # Counters and memory
    ctr: int = 0
    arms: jnp.ndarray = None
    rewards: jnp.ndarray = None


@struct.dataclass
class LGPUCB(BaseEstimator):
    # Fixed variables
    rkhs_norm_ub: float
    kappa: float
    L: float
    horizon: int
    duelling: bool
    solver: dict
    algo_name: str = "LGP-UCB"

    @classmethod
    def create(
        cls,
        domain: DiscreteDomain,
        rkhs_norm_ub: float,
        horizon: int,
        solver: dict = {"name": "GD", "args": {}},
        duelling: bool = False,
    ):
        if not isinstance(domain, DiscreteDomain):  # Other domains are not supported
            raise NotImplementedError("Only DiscreteDomain is supported")
        if duelling:
            kappa = 1 / dsigmoid(2 * rkhs_norm_ub)
        else:
            kappa = 1 / dsigmoid(rkhs_norm_ub)
        L = dsigmoid(0)
        return cls(
            domain=domain,
            rkhs_norm_ub=rkhs_norm_ub,
            kappa=kappa,
            L=L,
            horizon=horizon,
            solver=solver,
            duelling=duelling,
        )

    @property
    def default_params(self):
        return LGPUCBParams()

    def reset(self, rng: PRNGKey, params: LGPUCBParams):
        params = params.replace(
            alpha=jnp.full(self.horizon, jnp.nan),
            gram_matrix=jnp.full((self.horizon, self.horizon), jnp.nan),
            # gram_matrix_L=jnp.full((self.horizon, self.horizon), jnp.nan),
            ctr=0,
            arms=jnp.full((self.horizon, 2), jnp.nan) if self.duelling else jnp.full(self.horizon, jnp.nan),
            rewards=jnp.full(self.horizon, jnp.nan),
        )
        return params

    def _update_alpha(
        self,
            rng: PRNGKey,
            estimator_params: LGPUCBParams,
            value_and_grad: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        K = estimator_params.gram_matrix  # Shape: (horizon, horizon)
        # Update the alpha coefficients by minimizing the cross-entropy MLE
        K_filled_0 = jnp.where(jnp.isnan(K), 0, K)
        mask_in_horizon = ~jnp.isnan(estimator_params.rewards)
        rewards_filled_0 = jnp.where(mask_in_horizon, estimator_params.rewards, 0)
        rewards_filled_1 = jnp.where(mask_in_horizon, estimator_params.rewards, 1)

        def negative_log_likelihood(
            alpha: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Maximum likelihood estimation of the cross-entropy.
            Potential NaN values are filled such that it does not affect the gradient flow and the result
            """
            logit = jnp.dot(K_filled_0, alpha)  # Shape: (horizon,)
            probs = jax.nn.sigmoid(logit)
            alpha_masked = jnp.where(mask_in_horizon, alpha, 0)
            # Negative log-likelihood
            nll = -rewards_filled_0 * jnp.log(jnp.clip(probs, a_min=1e-36)) - (
                1 - rewards_filled_1
            ) * jnp.log(jnp.clip(1 - probs, a_min=1e-36))
            nll = jnp.where(mask_in_horizon, nll, 0)
            nll = jnp.sum(nll)
            # Weight regularization
            alpha_masked_norm = jnp.dot(alpha_masked, alpha_masked)
            regularization = (
                estimator_params.nll_regularization_penalty * alpha_masked_norm
            )
            # Gradient
            error = jnp.where(
                mask_in_horizon, probs - rewards_filled_0, 0
            )
            return nll + regularization

        if self.solver["name"] == "GD":
            solver = jaxopt.GradientDescent(
                fun=negative_log_likelihood,
                value_and_grad=value_and_grad,
                **self.solver["args"],
            )
        elif self.solver["name"] == "LBFGS":
            solver = jaxopt.LBFGS(
                fun=negative_log_likelihood,
                value_and_grad=value_and_grad,
                **self.solver["args"],
            )
        else:
            raise NotImplementedError(f"Solver {self.solver['name']} is not implemented")
        init_alpha = jax.random.normal(rng, shape=(self.horizon,))
        init_alpha = jnp.where(mask_in_horizon, init_alpha, 0)
        init_alpha = (
            self.rkhs_norm_ub * init_alpha / jnp.linalg.norm(init_alpha, 2)
        )  # Normalize
        results = solver.run(init_alpha)
        alpha = jnp.where(
            mask_in_horizon, results.params, jnp.nan
        )  # Replace unused values with NaNs

        # Update the parameters
        estimator_params = estimator_params.replace(
            alpha=alpha,
        )
        info = {
            "nll": results.state.value,
            "alpha": alpha,
        }
        return estimator_params, info

    def _get_arm_feature(self, arm: Union[int, jnp.ndarray]) -> jnp.ndarray:
        return self.domain.get_feature(arm)

    def update(
        self,
        key: PRNGKey,
        arm: Union[int, Tuple[int, int]],
        feedback: int,
        params: LGPUCBParams,
    ) -> Tuple[LGPUCBParams, Dict[str, jnp.ndarray]]:
        """Update the parameters of the algorithm."""
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
                self.domain.get_feature(arm[1])
            )  # Tuple of 2, Shape: (dim,)
        else:
            history_feature_matrix = self._get_arm_feature(arms_updated)
            arm_feature = self._get_arm_feature(arm)
        cross_variance = params.kernel.cross_covariance(
            history_feature_matrix, arm_feature
        )[
            :, 0
        ]  # Shape: (horizon,)
        gram_matrix_updated = params.gram_matrix.at[params.ctr, :].set(cross_variance)
        gram_matrix_updated = gram_matrix_updated.at[:, params.ctr].set(cross_variance)

        params = params.replace(
            arms=arms_updated,
            rewards=rewards_updated,
            gram_matrix=gram_matrix_updated,
            ctr=params.ctr + 1,
        )

        params, info = jax.lax.cond(
            params.ctr > 0,
            self._update_alpha,
            lambda key, x: (
                x,
                {"nll": jnp.nan, "alpha": jnp.full_like(x.rewards, jnp.nan)},
            ),  # Do nothing
            key,
            params,
        )
        return params, info

    def _calculate_beta(
        self, params: LGPUCBParams, covariance_matrix_L: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the beta value for the UCB algorithm.
            beta = 4LB + 2L/sqrt(lambda*kappa) * sqrt(2log(1/delta) + logdet(covariance_matrix))
        The log determinant is calculated from the Cholesky decomposition of the covariance matrix.
        :param params:
        :param covariance_matrix:
        :return:
        """
        cov_logdet = jnp.where(
            jnp.isnan(params.rewards),
            0.0,
            jnp.log(jnp.diag(covariance_matrix_L)),
        )
        cov_logdet = 2 * jnp.sum(cov_logdet)
        B = self.rkhs_norm_ub
        rho = jnp.sqrt(self.kappa * params.lambda_)
        L = self.L
        beta = 4 * L * B + (2 * L) / rho * jnp.sqrt(
            2 * jnp.log(1 / params.delta) + cov_logdet
        )
        return beta

    def estimate(
        self, params: LGPUCBParams
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the mean and variance of the posterior distribution."""
        if self.duelling:
            history_feature_matrix = (
                self.domain.get_feature(params.arms[:, 0]),
                self.domain.get_feature(params.arms[:, 1]),
            )  # Tuple of 2, Shape: (horizon, dim), with NaNs
        else:
            history_feature_matrix = self._get_arm_feature(params.arms)  # Shape: (horizon, dim), with NaNs
        feature_matrix = self.domain.feature_matrix  # Shape: (n_arms, dim)

        # Calculate the covariance matrix
        K = params.gram_matrix  # Shape: (horizon, horizon)
        K_filled_0 = jnp.where(jnp.isnan(K), 0, K)
        rho2 = self.kappa * params.lambda_
        covariance_matrix = K_filled_0 + rho2 * jnp.eye(self.horizon)
        L = jnp.linalg.cholesky(covariance_matrix)  # Shape: (horizon, horizon)
        L = L - jnp.where(
            jnp.isnan(params.rewards), jnp.sqrt(rho2), 0.0
        )  # Set diagonals beyond ctr to zero

        # Calculate the posterior mean and variance
        if self.duelling:
            duelling_cross_variance_vmap = jax.vmap(
                jax.vmap(
                    lambda x, y: params.kernel.cross_covariance(history_feature_matrix, (x, y))[:, 0],  # Shape: (horizon, )
                    in_axes=(None, 0),
                    out_axes=1,
                ),  # Shape: (horizon, n_arms (y))
                in_axes=(0, None),
                out_axes=1,
            )  # Shape: (horizon, n_arms (x), n_arms (y))
            duelling_self_gram_vmap = jax.vmap(
                jax.vmap(
                    lambda x, y: params.kernel.gram((x, y))[0, 0],  # Shape: (, )
                    in_axes=(None, 0),
                ),  # Shape: (n_arms (y), )
                in_axes=(0, None),
            )  # Shape: (n_arms (x), n_arms (y))

            cross_variance = duelling_cross_variance_vmap(feature_matrix, feature_matrix)  # Shape: (horizon, n_arms, n_arms)
            self_gram = duelling_self_gram_vmap(feature_matrix, feature_matrix)  # Shape: (n_arms, n_arms)
        else:
            cross_variance = params.kernel.cross_covariance(history_feature_matrix, feature_matrix)  # Shape: (horizon, n_arms)
            self_gram = jax.vmap(params.kernel.gram)(feature_matrix)[:, 0, 0]  # Shape: (n_arms, )

        # Posterior mean and variance
        def posterior_variance(k: jnp.ndarray, self_k: jnp.ndarray) -> jnp.ndarray:
            v = jax.scipy.linalg.solve_triangular(L, k, lower=True)
            variance = self_k - jnp.nansum(v**2)
            return variance

        if self.duelling:
            posterior_mean = jax.nn.sigmoid(
                jnp.nansum(cross_variance * params.alpha[:, None, None], axis=0)
            )  # Shape: (n_arms, n_arms)
            posterior_var = jax.vmap(
                    jax.vmap(
                    posterior_variance,
                    in_axes=(1, 0)
                ),
                in_axes=(1, 0),
            )(
                jnp.nan_to_num(cross_variance, nan=0.0), self_gram
            )  # Shape: (n_arms, n_arms)
        else:
            posterior_mean = jax.nn.sigmoid(
                jnp.nansum(cross_variance * params.alpha[:, None], axis=0)
            )  # Shape: (n_arms,)
            posterior_var = jax.vmap(posterior_variance, in_axes=(1, 0))(
                jnp.nan_to_num(cross_variance, nan=0.0), self_gram
            )  # Shape: (n_arms,)

        # Calculate the beta
        beta = jax.lax.cond(
            jnp.isnan(params.beta),
            self._calculate_beta,
            lambda params, L: params.beta,
            params,
            L,
        )

        return posterior_mean, posterior_var, beta

    def best_arm(
        self, key: PRNGKey, params: LGPUCBParams
    ) -> Tuple[int, float, LGPUCBParams]:
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
