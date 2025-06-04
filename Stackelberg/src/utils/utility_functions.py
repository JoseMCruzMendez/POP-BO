from flax import struct
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Union
from chex import PRNGKey
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV, cross_validate, train_test_split

from src.environments.Domain.DiscreteDomain import DiscreteDomain
from src.environments.Domain.ContinuousDomain import ContinuousDomain

"""
UTILITY FUNCTIONS
"""


def scale_theta(theta: jax.Array, norm_ub: float) -> jax.Array:
    """Scales the parameters to the given norm upper bound."""
    if norm_ub is None:
        return theta
    else:
        return norm_ub * theta / jnp.linalg.norm(theta, 2)


def get_features_for_discrete_domain(arm: jax.Array, domain: DiscreteDomain):
    """
    Get the feature representation of the arm for a discrete domain.
    If no features are provided, one-hot encode the arm.
    :param arm: Array with the arm index.
    :param domain: DiscreteDomain object.
    :return: arm_feature: Array with the feature representation of the arm.
    """
    if domain.has_features:
        arm_feature = domain.get_feature(arm)
    else:
        arm_feature = jax.nn.one_hot(arm, domain.num_elements)
    return arm_feature


def normalize_affine_utility(
    domain: Union[DiscreteDomain, ContinuousDomain],
    utility_function: Callable,
    utility_params: struct.dataclass,
    output_range: Tuple[float, float] = jnp.nan,
) -> struct.dataclass:
    """
    Normalize parameters of the utility function s.t. the maximum utility is equal to max_utility over the domain.
    """

    def f(arm: jax.Array) -> jax.Array:
        return utility_function(arm, utility_params)

    max_utility_over_domain = domain.maximize(f)[1]
    min_utility_over_domain = -domain.maximize(lambda x: -f(x))[1]
    scaling_factor = (output_range[1] - output_range[0]) / (
        max_utility_over_domain - min_utility_over_domain
    )
    affine_transform = jax.lax.select(
        jnp.isnan(scaling_factor),
        jnp.array([1.0, 0.0]),
        jnp.array(
            [
                scaling_factor,
                -scaling_factor * min_utility_over_domain + output_range[0],
            ]
        ),
    )
    utility_params = utility_params.replace(affine_transform=affine_transform)
    return utility_params


"""
LINEAR & POLIYNOMIAL UTILITY
"""


def create_linear_utility(
    rng: PRNGKey, domain: Union[DiscreteDomain, ContinuousDomain], function_config: dict
) -> Tuple[struct.dataclass, Callable[[jax.Array, struct.dataclass], jax.Array]]:
    """Creates a linear utility function and its parameters."""

    @struct.dataclass
    class LinearUtilityFunctionParams:
        theta: jax.Array
        bias: float
        affine_transform: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0]))

    def linear_utility(
        arm: jax.Array, params: LinearUtilityFunctionParams
    ) -> jax.Array:
        if isinstance(domain, DiscreteDomain):
            arm = get_features_for_discrete_domain(arm, domain)
        return jnp.dot(arm, params.theta) + params.bias

    if function_config["param_initialization"] == "normal":
        theta = jax.random.normal(rng, shape=(domain.feature_dim,))
        theta = scale_theta(theta, function_config["param_norm_ub"])
    else:
        raise NotImplementedError
    utility_params = LinearUtilityFunctionParams(
        theta=theta, bias=function_config["bias"]
    )
    utility_params = normalize_affine_utility(
        domain,
        linear_utility,
        utility_params,
        function_config["utility_range"],
    )
    return utility_params, linear_utility


def create_polynomial_utility(
    rng: PRNGKey, domain: Union[DiscreteDomain, ContinuousDomain], function_config: dict
) -> Tuple[struct.dataclass, Callable[[jax.Array, struct.dataclass], jax.Array]]:
    """Creates a polynomial utility function and its parameters."""

    @struct.dataclass
    class PolynomialUtilityFunctionParams:
        order: int
        theta: jax.Array  # Shape: (order, feature_dim)
        bias: float
        affine_transform: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0]))

    def polynomial_utility(
        arm: jax.Array, params: PolynomialUtilityFunctionParams
    ) -> jax.Array:
        if isinstance(domain, DiscreteDomain):
            arm = get_features_for_discrete_domain(arm, domain)
        arm_poly = jnp.power(
            arm[None, :], jnp.arange(1, params.order + 1)[:, None]
        )  # Shape: (order, feature_dim)
        utility = jnp.sum(arm_poly * params.theta) + params.bias
        utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
        return utility

    if function_config["param_initialization"] == "normal":
        theta = jax.random.normal(
            rng,
            shape=(function_config["poly_degree"], domain.feature_dim)
            # shape=(domain.feature_dim,),
        )
        theta = scale_theta(theta, function_config["param_norm_ub"])
    else:
        raise NotImplementedError
    utility_params = PolynomialUtilityFunctionParams(
        theta=theta, bias=function_config["bias"], order=function_config["poly_degree"]
    )
    utility_params = normalize_affine_utility(
        domain,
        polynomial_utility,
        utility_params,
        function_config["utility_range"],
    )
    return utility_params, polynomial_utility


"""
STANDARD FUNCTIONS FOR OPTIMISATION
https://arxiv.org/pdf/physics/0402085.pdf
https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-3

NOTE: functions are negated to convert minimisation to maximisation
"""


@struct.dataclass
class AffineTransformFunctionParams:
    affine_transform: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0]))


def ackley(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = 20*exp(-0.2*sqrt(||x||_2 / n)) + exp(sum(cos(2*pi*x)) / n) - 20 - exp(1)
    supported on x \in [-32.768, 32.768]
    """
    n = arm.shape[0]
    utility = (
        20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(arm**2) / n))
        + jnp.exp(jnp.sum(jnp.cos(2 * jnp.pi * arm)) / n)
        - 20
        - jnp.exp(1)
    )
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def branin(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = -(a*(y - b*x^2 + c*x - r)^2 + s*(1 - t)*cos(x) + s)
    supported on x \in [-5, 10], y \in [0, 15]
    """
    a, b, c, r, s, t = 1, 5.1 / (4 * jnp.pi**2), 5 / jnp.pi, 6, 10, 1 / (8 * jnp.pi)
    assert arm.shape[0] == 2
    x, y = arm
    utility = -(a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * jnp.cos(x) + s)
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def eggholder(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = (y + 47)*sin(sqrt(abs(y + x/2 + 47))) + x*sin(sqrt(abs(x - (y + 47))))
    supported on x \in [-512, 512], y \in [-512, 512]
    """
    # assert that the arm has 2 dimensions
    assert arm.shape[0] == 2
    x, y = arm
    utility = (y + 47) * jnp.sin(jnp.sqrt(jnp.abs(y + x / 2 + 47))) + x * jnp.sin(
        jnp.sqrt(jnp.abs(x - (y + 47)))
    )
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def hoelder(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = |sin(x)*cos(y)*exp(abs(1 - sqrt(x^2 + y^2)/pi))|
    supported on x \in [-10, 10], y \in [-10, 10]
    """
    # assert that the arm has 2 dimensions
    assert arm.shape[0] == 2
    x, y = arm
    utility = jnp.abs(
        jnp.sin(x)
        * jnp.cos(y)
        * jnp.exp(jnp.abs(1 - jnp.sqrt(x**2 + y**2) / jnp.pi))
    )
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def matyas(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = -(0.26*(x^2 + y^2) - 0.48*x*y)
    supported on x \in [-10, 10], y \in [-10, 10]
    """
    # assert that the arm has 2 dimensions
    assert arm.shape[0] == 2
    x, y = arm
    utility = -(0.26 * (x**2 + y**2) - 0.48 * x * y)
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def michalewicz(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = sum(sin(x_i)*sin(i*x_i^2/pi)^10)
    supported on x \in [0, pi]
    """
    utility = jnp.sum(
        jnp.sin(arm)
        * jnp.sin(jnp.arange(1, arm.shape[0] + 1) * arm**2 / jnp.pi) ** 10
    )
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def rosenbrock(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = -sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    supported on x \in [-5, 10]
    """
    utility = -jnp.sum(100 * (arm[1:] - arm[:-1] ** 2) ** 2 + (1 - arm[:-1]) ** 2)
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def bukin(arm: jax.Array, params: AffineTransformFunctionParams) -> jax.Array:
    """
    u(x) = -100*sqrt(abs(y - 0.01*x^2)) - 0.01*abs(x + 10)
    supported on x \in [-15, -5], y \in [-3, 3]
    """
    # assert that the arm has 2 dimensions
    assert arm.shape[0] == 2
    x, y = arm
    utility = -100 * jnp.sqrt(jnp.abs(y - 0.01 * x**2)) - 0.01 * jnp.abs(x + 10)
    utility = jnp.dot(params.affine_transform, jnp.array([utility, 1.0]))
    return utility


def create_standard_optimisation_function(
    rng: PRNGKey,
    domain: Union[DiscreteDomain, ContinuousDomain],
    function_config: dict,
    utility_function_name: str,
) -> Tuple[struct.dataclass, Callable[[jax.Array, struct.dataclass], jax.Array]]:
    if utility_function_name == "ackley":
        utility_function = ackley
    elif utility_function_name == "hoelder":
        utility_function = hoelder
    elif utility_function_name == "eggholder":
        utility_function = eggholder
    elif utility_function_name == "rosenbrock":
        utility_function = rosenbrock
    elif utility_function_name == "bukin":
        utility_function = bukin
    elif utility_function_name == "branin":
        utility_function = branin
    elif utility_function_name == "michalewicz":
        utility_function = michalewicz
    elif utility_function_name == "matyas":
        utility_function = matyas
    else:
        raise NotImplementedError

    def utility_function_for_domain(arm: jax.Array, utility_params) -> jax.Array:
        if isinstance(domain, DiscreteDomain):
            arm = get_features_for_discrete_domain(arm, domain)
        return utility_function(arm, utility_params)

    utility_params = AffineTransformFunctionParams()
    utility_params = normalize_affine_utility(
        domain,
        utility_function_for_domain,
        utility_params,
        function_config["utility_range"],
    )
    return utility_params, utility_function_for_domain


"""
YELP DATASET
"""


def load_yelp_data(
    data_dir: str,
    city: str = "Philadelphia",
    min_review_count: int = 500,
    min_review_per_user: int = 90,
    collaborative_filtering_args: dict = {
        "n_factors": 30,
        "n_epochs": 50,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
) -> Dict[str, jax.Array]:
    """
    Load Yelp data and return arm features and utilities.
    :param city:
    :param min_review_count:
    :param min_review_per_user:
    :param collaborative_filtering_args:
    :return:
        - arm_features: Array with the arm features. Shape: (num_arms, feature_dim)
        - utilities: Array with the utilities. Shape: (num_users, num_arms)
    """
    df_path = f"{data_dir}/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}.csv"
    business_aggregated_reviews = f"{data_dir}/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}_aggregated.csv"
    embeddings_path = f"{data_dir}/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}_embeddings_32.npy"

    df = pd.read_csv(df_path, index_col=0)
    df_reviews_aggregated = pd.read_csv(business_aggregated_reviews, index_col=0)
    embeddings = np.load(embeddings_path)

    # Collaborative filtering to estimate all user ratings
    df_user_business_avg_stars = (
        df.sort_values("date")
        .groupby(["user_id", "business_id"])["stars"]
        .last()
        .reset_index()
    )
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_user_business_avg_stars, reader)
    model = SVD(**collaborative_filtering_args)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    all_user_business_pairs = [
        (user_id, business_id) for user_id in df["user_id"].unique() for business_id in df["business_id"].unique()
    ]
    df_pred = [model.predict(user_id, business_id) for user_id, business_id in all_user_business_pairs]
    df_pred = pd.DataFrame(df_pred)
    df_pred.columns = ["user_id", "business_id", "actual_rating", "pred_rating", "details"]
    utilities = df_pred.set_index(["user_id", "business_id"])["pred_rating"].unstack().loc[:, df_reviews_aggregated.index].values
    utilities = jnp.array(utilities)
    return {"arm_features": embeddings, "utilities": utilities}


def create_yelp_utility(
    rng: PRNGKey,
    domain: DiscreteDomain,
    function_config: dict,
    utilities_arr: jax.Array,
) -> Tuple[struct.dataclass, Callable[[int, struct.dataclass], jax.Array]]:
    """Creates a polynomial utility function and its parameters."""
    assert domain.num_elements == utilities_arr.shape[1]

    # Randomly select one user
    user_idx = jax.random.randint(rng, shape=(), minval=0, maxval=utilities_arr.shape[0]-1)
    # user_utilities = utilities_arr[user_idx, :]
    user_utilities = jax.lax.dynamic_index_in_dim(
        utilities_arr, user_idx, axis=0, keepdims=False
    )

    @struct.dataclass
    class YelpUtilityFunctionParams:
        utilities: jax.Array  # Shape: (num_arms,)
        affine_transform: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0]))

    def yelp_utility(arm: int, params: YelpUtilityFunctionParams) -> jax.Array:
        return params.utilities[arm]

    utility_params = YelpUtilityFunctionParams(
        utilities=user_utilities,
        affine_transform=jnp.array([1.0, 0.0])
    )
    utility_params = normalize_affine_utility(
        domain,
        yelp_utility,
        utility_params,
        function_config["utility_range"],
    )
    return utility_params, yelp_utility
