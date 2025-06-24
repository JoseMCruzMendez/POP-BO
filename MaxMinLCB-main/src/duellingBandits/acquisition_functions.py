import jax
import jax.numpy as jnp
from chex import PRNGKey

from typing import Tuple
from functools import partial


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer", "use_candidate_set"])
def max_min_lcb(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    use_candidate_set: bool = True,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    lcb = posterior_mean - beta * posterior_var  # Shape: (n_arms, n_arms)
    n = lcb.shape[0]

    # Set values to nan for arms that are clearly suboptimal
    if use_candidate_set:
        ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
        candidate_arms_mask = jnp.all(
            jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
            axis=1,
        )  # Shape: (n_arms,)
        # Make sure you do not consider the same arms at once, Shape: (n_arms, )
        lcb = jnp.where(
            candidate_arms_mask[:, None] * candidate_arms_mask[None, :],
            lcb,
            jnp.nan,
        )
        lcb = jnp.where(jnp.eye(lcb.shape[0]), jnp.nan, lcb)
    else:
        candidate_arms_mask = jnp.ones(n, dtype=bool)
        lcb = jnp.where(jnp.eye(n), 0.5, lcb)

    min_j = jnp.nanmin(lcb, axis=1)  # Shape: (n_arms, )
    # argmin_j = jnp.nanargmin(lcb, axis=1)  # Shape: (n_arms, )
    argmin_j_set = jnp.where(
        jnp.abs(lcb - min_j[:,None]) < argmax_tol,
        jax.random.choice(key, n**2, shape=(n, n), replace=False),
        -jnp.inf,
    )
    argmin_j = jnp.argmax(argmin_j_set, axis=1)
    maxmin_lcb = jnp.nanmax(min_j)  # Shape: ()

    def choose_next_arms():
        argmax_set = jnp.where(
            jnp.abs(min_j - maxmin_lcb) < argmax_tol,
            jax.random.choice(key, n, shape=(n,), replace=False),
            jnp.nan,
        )
        next_arm_i = jnp.nanargmax(argmax_set)
        next_arm_j = argmin_j[next_arm_i]
        return next_arm_i, next_arm_j, acquisition_info

    return jax.lax.cond(
        jnp.sum(candidate_arms_mask) == 1,
        lambda: (
            jnp.nanargmax(candidate_arms_mask),
            jnp.nanargmax(candidate_arms_mask),
            acquisition_info
        ),
        choose_next_arms,
    )


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def min_max_ucb(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Min-Max UCB acquisition function
    Note that it is equivalent to max_min_lcb but with reverse order of arm selection
    """
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    n = ucb.shape[0]

    # Set values to nan for arms that are clearly suboptimal
    candidate_arms_mask = jnp.all(
        jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
        axis=1,
    )  # Shape: (n_arms,)
    # Make sure you do not consider the same arms at once, Shape: (n_arms, )
    ucb = jnp.where(jnp.eye(n), jnp.nan, ucb)

    max_j = jnp.nanmax(ucb, axis=1)  # Shape: (n_arms, )
    argmax_j = jnp.nanargmin(ucb, axis=1)  # Shape: (n_arms, )
    minmax_ucb = jnp.nanmin(max_j)  # Shape: ()

    def choose_next_arms():
        argmax_set = jnp.where(
            jnp.abs(max_j - minmax_ucb) < argmax_tol,
            jax.random.choice(key, n, shape=(n,), replace=False),
            jnp.nan,
        )
        next_arm_i = jnp.nanargmax(argmax_set)
        next_arm_j = argmax_j[next_arm_i]
        return next_arm_i, next_arm_j, acquisition_info

    return jax.lax.cond(
        jnp.sum(candidate_arms_mask) == 1,
        lambda: (
            jnp.nanargmax(candidate_arms_mask),
            jnp.nanargmax(candidate_arms_mask),
            acquisition_info
        ),
        choose_next_arms,
    )


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def max_max_ucb(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    ucb = jnp.where(jnp.eye(ucb.shape[0]), jnp.nan, ucb)
    n = ucb.shape[0]
    argmax_j, max_j = jnp.nanargmax(ucb, axis=1), jnp.nanmax(ucb, axis=1)

    argmax_set = jnp.where(
        jnp.abs(max_j - jnp.max(max_j)) < argmax_tol,
        jax.random.choice(key, n, shape=(n,), replace=False),
        jnp.nan,
    )
    next_arm_i = jnp.nanargmax(argmax_set)
    next_arm_j = jax.lax.select(
        max_j[next_arm_i]
        < 0.5 - decision_buffer,  # If the UCB is lower than 0.5, choose itself
        next_arm_i,
        argmax_j[next_arm_i],
    )
    return next_arm_i, next_arm_j, acquisition_info


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def max_info_gain(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    n = ucb.shape[0]
    candidate_arms_mask = jnp.all(
        jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
        axis=1,
    )  # Shape: (n_arms,)

    posterior_var_masked = jnp.where(
        candidate_arms_mask[:, None] & candidate_arms_mask[None, :],
        posterior_var,
        -jnp.inf,
    ) - jnp.inf * jnp.eye(n)

    def choose_next_arm():
        first_candidate = jnp.argmax(candidate_arms_mask)
        argmax_set = jnp.where(
            jnp.abs(posterior_var_masked - jnp.max(posterior_var_masked)) < argmax_tol,
            jax.random.choice(key, n**2, shape=(n, n), replace=False),
            -jnp.inf,
        )
        max_idx = jnp.argmax(argmax_set)
        max_idx = jnp.unravel_index(max_idx, (n, n))

        # If only one candidate arm is available, choose it
        return jax.lax.cond(
            jnp.sum(candidate_arms_mask) == 1,
            lambda: (first_candidate, first_candidate, acquisition_info),
            lambda: (max_idx[0], max_idx[1], acquisition_info),
        )

    return jax.lax.cond(
        jnp.isinf(jnp.max(posterior_var_masked)),
        lambda: (jnp.argmax(candidate_arms_mask), jnp.argmax(candidate_arms_mask), acquisition_info),
        choose_next_arm,
    )


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def MultiSBM(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = {"previous_arm": -1},
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    n = posterior_mean.shape[0]
    previous_arm = acquisition_info["previous_arm"]

    def choose_random_arms():
        next_arms = jax.random.choice(key, n, shape=(2,), replace=False)
        return next_arms[0], next_arms[1], {"previous_arm": next_arms[1]}

    def choose_next_arm():
        ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
        next_arm_i = previous_arm
        is_next_arm_mask = jnp.arange(n) == next_arm_i
        ucb = jnp.where(
            (~is_next_arm_mask)[:, None] * is_next_arm_mask[None, :], ucb, -jnp.inf
        )  # Shape: (n_arms, n_arms)
        max_ucb_i = jnp.max(ucb)

        def choose_argmax():
            argmax_set = jnp.where(
                jnp.abs(ucb - max_ucb_i) < argmax_tol,
                jax.random.choice(key, n**2, shape=(n, n), replace=False),
                -jnp.inf,
            )
            max_idx = jnp.argmax(argmax_set)
            max_idx = jnp.unravel_index(max_idx, (n, n))
            return max_idx[0]

        next_arm_j = jax.lax.cond(
            max_ucb_i < 0.5 - decision_buffer,  # If the UCB is lower than 0.5, choose itself
            lambda: next_arm_i,
            choose_argmax,
        )
        return next_arm_i, next_arm_j, {"previous_arm": next_arm_j}

    return jax.lax.cond(previous_arm < 0, choose_random_arms, choose_next_arm)


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer", "prob_grid_size"])
def information_directed_sampling(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    prob_grid_size: int = 100,
    rho2: float = 1.0,
):
    # Find feasible set
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    n = ucb.shape[0]
    candidate_arms_mask = jnp.all(
        jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
        axis=1,
    )  # Shape: (n_arms,)

    # Choose argmax
    null_arm_idx = 0
    max_greedy_value = jnp.max(
        jnp.where(
            candidate_arms_mask,
            posterior_mean[:, null_arm_idx],
            -jnp.inf,
        )
    )
    key, _rng = jax.random.split(key)
    greedy_set = jnp.where(
        jnp.logical_and(
            jnp.abs(posterior_mean[:, null_arm_idx] - max_greedy_value) < argmax_tol,
            candidate_arms_mask,
        ),
        jax.random.choice(
            _rng,
            posterior_mean.shape[0],
            shape=(posterior_mean.shape[0],),
            replace=False,
        ),
        -jnp.inf,
    )
    greedy_arm_idx = jnp.argmax(greedy_set)
    greedy_arm_idx = jax.lax.select(
        jnp.logical_and(
            posterior_mean[greedy_arm_idx, null_arm_idx] < 0.5,
            candidate_arms_mask[null_arm_idx]
        ),
        null_arm_idx,
        greedy_arm_idx,
    )

    # Define utility upperbound
    max_reward = jnp.max(jnp.where(candidate_arms_mask, ucb[:, greedy_arm_idx], -jnp.inf))

    # Define suboptimality gap
    suboptimality_gap = (
        max_reward + posterior_mean[greedy_arm_idx, :]
    )  # Shape: (n_arms, )

    # Calculate ids criteria
    prob_grid = jnp.linspace(0, 1, prob_grid_size + 1)[1:].reshape(
        1, -1
    )  # Cut first entry to avoid division by 0, Shape: (1, prob_grid_size)
    ids = (
        jnp.power(
            (1 - prob_grid) * max_reward + prob_grid * suboptimality_gap.reshape(-1, 1),
            2,
        )
        / prob_grid
        * jnp.log(1 + posterior_var[greedy_arm_idx, :].reshape(-1, 1) / rho2)
    )  # Shape: (n_arms, prob_grid_size)
    ids = ids.at[greedy_arm_idx, :].set(jnp.inf)
    ids = jnp.where(candidate_arms_mask.reshape(-1, 1), ids, jnp.inf)

    # Optimise for arm and p
    key, _rng = jax.random.split(key)
    ids_min_set = jnp.where(
        jnp.abs(ids - jnp.min(ids)) < argmax_tol,
        jax.random.choice(
            _rng, ids.shape[0] * ids.shape[1], shape=ids.shape, replace=False
        ),
        -jnp.inf,
    )
    next_arm_j, prob_idx = jnp.unravel_index(jnp.argmax(ids_min_set), ids.shape)

    # Return next arms
    next_arm_i, next_arm_j = jax.lax.cond(
        jax.random.bernoulli(key, prob_grid[0, prob_idx]),
        lambda: (greedy_arm_idx, next_arm_j),
        lambda: (greedy_arm_idx, greedy_arm_idx),
    )

    return jax.lax.cond(
        jnp.sum(candidate_arms_mask) == 1,
        lambda: (
            jnp.nanargmax(candidate_arms_mask),
            jnp.nanargmax(candidate_arms_mask),
            acquisition_info
        ),
        lambda: (next_arm_i, next_arm_j, acquisition_info),
    )


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer", "use_candidate_set"])
def Doubler(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    use_candidate_set: bool = True,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:

    # Find feasible set
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    n = ucb.shape[0]
    if use_candidate_set:
        candidate_arms_mask = jnp.all(
            jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
            axis=1,
        )  # Shape: (n_arms,)
    else:
        candidate_arms_mask = jnp.ones(n, dtype=bool)

    # Choose first arm randomly from the feasible set
    key, _rng = jax.random.split(key)
    next_arm_j = jax.random.choice(
        _rng,
        n,
        shape=(1,),
        replace=False,
        p=jnp.where(candidate_arms_mask, 1/jnp.sum(candidate_arms_mask), 0),
    )[0]

    ucb = jnp.where(jnp.eye(n), jnp.nan, ucb)
    ucb_mean = jnp.where(
        candidate_arms_mask,
        jnp.nanmean(ucb, axis=1),
        -jnp.inf,
    )
    max_ucb_mean = jnp.max(ucb_mean)
    ucb_mean_max_set = jnp.where(
        jnp.abs(ucb_mean - max_ucb_mean) < argmax_tol,
        jax.random.choice(key, n, shape=(n,), replace=False),
        -jnp.inf,
    )
    next_arm_i = jnp.argmax(ucb_mean_max_set)

    acquisition_info["previous_arm"] = next_arm_i
    return next_arm_i, next_arm_j, acquisition_info


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer", "use_candidate_set"])
def Doubler_single_estimator(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = {},
    argmax_tol=1e-4,
    decision_buffer=0.0,
    use_candidate_set: bool = True,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    i, j, selection_set, next_set = acquisition_info["i"], acquisition_info["j"], acquisition_info["selection_set"], acquisition_info["next_set"]
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms,)
    n = ucb.shape[0]

    # Choose uniformly from current_set
    key, _rng = jax.random.split(key)
    next_arm_j = jax.random.choice(
        _rng,
        n,
        shape=(1,),
        replace=False,
        p=selection_set/selection_set.sum(),
    )[0]

    # Choose the arm that maximizes the UCB mean
    max_ucb = jnp.max(ucb)
    ucb_max_set = jnp.where(
        jnp.abs(ucb - max_ucb) < argmax_tol,
        jax.random.choice(key, n, shape=(n,), replace=False),
        -jnp.inf,
    )
    next_arm_i = jnp.argmax(ucb_max_set)

    next_set = next_set.at[next_arm_i].set(next_set[next_arm_i] + 1)
    # next_set = next_set + jnp.eye(next_arm_i+1, n)
    i, j, selection_set, next_set = jax.lax.cond(
        j == 2 ** i,
        lambda _: (i + 1, 1, next_set, jnp.zeros(n,)),  # Next loop
        lambda _: (i, j + 1, selection_set, next_set),  # Update current
        None,
    )
    acquisition_info = {"i": i, "j": j, "selection_set": selection_set, "next_set": next_set}
    return next_arm_i, next_arm_j, acquisition_info


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def RUCB(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = None,
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Zoghi et al. (2014) - Relative upper confidence bound for the k-armed duelling bandit problem."""
    ucb = posterior_mean + beta * posterior_var  # Shape: (n_arms, n_arms)
    n = ucb.shape[0]
    candidate_arms_mask = jnp.all(
        jnp.logical_or(ucb > 0.5 - decision_buffer, jnp.diag(jnp.full(n, True))),
        axis=1,
    )  # Shape: (n_arms,)

    # Select next_arm_i randomly from candidate_arms_mask
    key, _rng = jax.random.split(key)
    next_arm_j = jax.random.choice(
        _rng,
        n,
        shape=(1,),
        replace=False,
        p=jnp.where(candidate_arms_mask, 1/jnp.sum(candidate_arms_mask), 0),
    )[0]

    # Select next_arm_j that maximizes the UCB given i
    ucb = jnp.where(jnp.eye(n), -jnp.inf, ucb)
    ucb_j = ucb[:, next_arm_j]
    max_ucb_j = jnp.max(ucb_j)
    ucb_j_max_set = jnp.where(
        jnp.abs(ucb_j - max_ucb_j) < argmax_tol,
        jax.random.choice(key, n, shape=(n,), replace=False),
        -jnp.inf,
    )
    next_arm_i = jnp.argmax(ucb_j_max_set)

    return jax.lax.cond(
        jnp.sum(candidate_arms_mask) == 1,
        lambda: (
            jnp.nanargmax(candidate_arms_mask),
            jnp.nanargmax(candidate_arms_mask),
            acquisition_info
        ),
        lambda: (next_arm_i, next_arm_j, acquisition_info),
    )

@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer"])
def Sparring(
    key: PRNGKey,
    posterior_mean: Tuple[jnp.ndarray, jnp.ndarray],
    posterior_var: Tuple[jnp.ndarray, jnp.ndarray],
    beta: Tuple[float, float],
    acquisition_info: dict = {},
    argmax_tol=1e-4,
    decision_buffer=0.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    # Choose two arm independently according to the estimated posterior UCBs
    next_arms = jnp.full(2, jnp.nan)
    for i, (mean, var, beta) in enumerate(zip(posterior_mean, posterior_var, beta)):
        ucb = mean + beta * var  # Shape: (n_arms,)
        n = ucb.shape[0]
        max_ucb = jnp.max(ucb)
        ucb_max_set = jnp.where(
            jnp.abs(ucb - max_ucb) < argmax_tol,
            jax.random.choice(key, n, shape=(n,), replace=False),
            -jnp.inf,
        )
        next_arms = next_arms.at[i].set(jnp.argmax(ucb_max_set))
    return next_arms[0], next_arms[1], acquisition_info


@partial(jax.jit, static_argnames=["argmax_tol", "decision_buffer", "max_sample_iter"])
def doubleTS(
    key: PRNGKey,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    beta: float,
    acquisition_info: dict = {},
    argmax_tol=1e-4,
    decision_buffer=0.0,
    max_sample_iter: int = 50,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    n = posterior_mean.shape[0]
    # Marginalize the posterior mean and variance over the second axis
    posterior_mean = jnp.mean(posterior_mean, axis=1)
    posterior_var = jnp.mean(posterior_var, axis=1)

    # Choose the first arm
    def ts_sampling(rng):
        ts = (
            posterior_mean
            + jax.random.uniform(rng, shape=posterior_mean.shape, minval=-1.0, maxval=1.0)
            * beta
            * posterior_var
        )
        return jnp.argmax(ts)
    key, _rng = jax.random.split(key)
    next_arm_i = jnp.argmax(_rng)

    # Choose the second arm using TS repeatedly until it is different from the first arm
    def loop_body(sampled_idx, _rng):
        return jax.lax.select(
            next_arm_i == sampled_idx,
            ts_sampling(_rng),
            sampled_idx,
        ), None
    rngs = jax.random.split(key, max_sample_iter)
    next_arm_j, _ = jax.lax.scan(loop_body, next_arm_i, rngs)
    next_arm_j = jax.lax.select(
        next_arm_i == next_arm_j,
        jax.random.choice(key, n),
        next_arm_j,
    )
    return next_arm_i, next_arm_j, acquisition_info

