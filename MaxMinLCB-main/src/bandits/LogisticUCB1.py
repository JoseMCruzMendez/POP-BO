from src.bandits.BaseEstimator import BaseEstimator
from src.environments.Domain.DiscreteDomain import DiscreteDomain

from src.utils.utils import sigmoid, dsigmoid, weighted_norm

from flax import struct
import jax.numpy as jnp
import jax
from chex import PRNGKey

from jax.scipy.stats import chi2

from typing import Tuple


@struct.dataclass
class UCB1Params:
    # Hyperparameters
    failure_level: float = 0.1
    lazy_update_fr: int = 1
    # Variables
    l2reg: float = None
    hessian_matrix: jnp.ndarray = None
    design_matrix: jnp.ndarray = None
    design_matrix_inv: jnp.ndarray = None
    theta_hat: jnp.ndarray = None
    ucb_bonus: jnp.ndarray = None
    # Counters and memory
    ctr: int = 0
    arms: jnp.ndarray = None
    rewards: jnp.ndarray = None


@struct.dataclass
class LogisticUCB1(BaseEstimator):
    algo_name: str = "LogUCB1"
    # Fixed variables
    param_norm_ub: float = None
    arm_norm_ub: float = None
    kappa: float = None
    horizon: int = None

    @classmethod
    def create(
        cls,
        domain: DiscreteDomain,
        param_norm_ub: float,
        arm_norm_ub: float,
        horizon: int,
    ):
        kappa = 1 / dsigmoid(param_norm_ub * arm_norm_ub)
        return cls(
            domain=domain,
            param_norm_ub=param_norm_ub,
            arm_norm_ub=arm_norm_ub,
            kappa=kappa,
            horizon=horizon,
        )

    @property
    def default_params(self):
        return UCB1Params()

    def reset(self, rng: PRNGKey, params: UCB1Params):
        dim = self.domain.feature_dim
        params = params.replace(
            l2reg=dim,
            hessian_matrix=dim * jnp.eye(dim),
            design_matrix=dim * jnp.eye(dim),
            design_matrix_inv=(1 / dim) * jnp.eye(dim),
            theta_hat=jax.random.normal(rng, (dim,)),
            ctr=0,
            arms=jnp.full(self.horizon, jnp.nan),
            rewards=jnp.full(self.horizon, jnp.nan),
        )
        ucb_bonus = self._get_ucb_bonus(params)
        params = params.replace(ucb_bonus=ucb_bonus)
        return params

    def _get_ucb_bonus(self, params: UCB1Params) -> float:
        """
        Calculates the ucb bonus function (slight refinement from the concentration result of Faury et al. 2020)
        """
        dim = self.domain.feature_dim
        _, logdet = jnp.linalg.slogdet(params.hessian_matrix)
        gamma = (
            jnp.log(1 / params.failure_level)
            - 0.5 * dim * jnp.log(params.l2reg)
            + jnp.log(
                chi2.cdf(2 * params.l2reg, df=dim) / chi2.cdf(params.l2reg, df=dim)
            )
            + 0.5 * logdet
        )
        gamma = jnp.min(
            jnp.array(
                [
                    jnp.sqrt(params.l2reg) / 2 + (2 / jnp.sqrt(params.l2reg)) * gamma,
                    1 + gamma,
                ]
            )
        )
        res = (
            0.25
            * jnp.sqrt(self.kappa)
            * jnp.min(
                jnp.array(
                    [
                        jnp.sqrt(1 + 2 * self.param_norm_ub) * gamma,
                        gamma + gamma**2 / jnp.sqrt(params.l2reg),
                    ]
                )
            )
        )
        res += jnp.sqrt(params.l2reg) * self.param_norm_ub
        return res

    def update(self, key: PRNGKey, arm: int, feedback: int, params: UCB1Params):
        params = params.replace(
            arms=params.arms.at[params.ctr].set(arm),
            rewards=params.rewards.at[params.ctr].set(feedback),
        )

        # learn the m.l.e by iterative approach (a few steps of Newton descent)
        dim = self.domain.feature_dim
        params = params.replace(l2reg=dim * jnp.log(2 + params.ctr))

        # if params.ctr % params.lazy_update_fr == 0 or len(params.rewards) < 200:  # TODO: Reimplement lazy evaluation
        # if lazy we learn with a reduced frequency
        theta_hat = params.theta_hat
        hessian = params.hessian_matrix
        arms_matrix = self.domain.get_feature(
            params.arms
        )  # Shape: (self.horizon, features_length), some rows are NaNs
        arms_matrix_zero_filled = jnp.nan_to_num(arms_matrix, nan=0.0)

        def gradient_descent_step(state, _):
            theta_hat, hessian = state
            coeffs = sigmoid(
                jnp.dot(arms_matrix, theta_hat)[:, None]
            )  # Shape: (horizon, 1), some rows are NaNs
            y = (
                coeffs - params.rewards[:, None]
            )  # Shape: (horizon, 1), some rows are NaNs
            grad = params.l2reg * theta_hat + jnp.nansum(
                y * arms_matrix, axis=0
            )  # Shape: (features_length,)
            coeffs_filled_zeros = jnp.nan_to_num(coeffs, nan=0.0)
            hessian = jnp.dot(
                arms_matrix_zero_filled.T,
                coeffs_filled_zeros
                * (1 - coeffs_filled_zeros)
                * arms_matrix_zero_filled,
            ) + params.l2reg * jnp.eye(dim)  # Shape: (features_length, features_length)
            theta_hat -= jnp.linalg.solve(hessian, grad)
            return (theta_hat, hessian), None

        (theta_hat, hessian), _ = jax.lax.scan(
            gradient_descent_step,
            (theta_hat, hessian),
            None,
            # length=num_gradient_update_steps  # TODO: Reimplement that parameter comes from params and avoid JAX error
            length=5,
        )

        params = params.replace(
            theta_hat=theta_hat,
            hessian_matrix=hessian,
        )
        # update counter
        params = params.replace(
            ctr=params.ctr + 1, ucb_bonus=self._get_ucb_bonus(params)
        )
        return params, None

    def best_arm(self, key: PRNGKey, params: UCB1Params) -> Tuple[int, float, UCB1Params]:
        # update bonus bonus
        params = params.replace(ucb_bonus=self._get_ucb_bonus(params))
        # select arm
        arm_ucb_values = jax.vmap(
            lambda arr: self.compute_optimistic_reward(arr, params)
        )(self.domain.feature_matrix)
        best_arm = jnp.argmax(arm_ucb_values)
        best_ucb_value = arm_ucb_values[best_arm]
        best_arm_feature = self.domain.get_feature(best_arm)
        # update design matrix and inverse
        design_matrix = params.design_matrix + jnp.outer(
            best_arm_feature, best_arm_feature
        )
        design_matrix_inv = params.design_matrix_inv - jnp.dot(
            params.design_matrix_inv,
            jnp.dot(
                jnp.outer(best_arm_feature, best_arm_feature), params.design_matrix_inv
            ),
        ) / (
            1
            + jnp.dot(
                best_arm_feature, jnp.dot(params.design_matrix_inv, best_arm_feature)
            )
        )
        params = params.replace(
            design_matrix=design_matrix, design_matrix_inv=design_matrix_inv
        )
        return best_arm, None, None, params

    @staticmethod
    def compute_optimistic_reward(
        arm_feature: jnp.ndarray, params: UCB1Params
    ) -> float:
        """
        Computes UCB for arm.
        """
        norm = weighted_norm(arm_feature, params.design_matrix_inv)
        pred_reward = sigmoid(jnp.sum(params.theta_hat * arm_feature))
        bonus = params.ucb_bonus * norm
        return pred_reward + bonus
