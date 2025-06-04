import jax.numpy as jnp
from typing import Union


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def weighted_norm(x: jnp.ndarray, A: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """
    Compute the weighted norm of x with respect to A
        weighted_norm = sqrt(x^T A x)
    """
    if len(x.shape) == 1:
        return jnp.sqrt(jnp.dot(x, jnp.dot(A, x)))
    elif len(x.shape) == 2:
        # return np.diag(np.sqrt(np.dot(x, np.dot(A, x.T))))
        return jnp.sqrt(jnp.dot(x, A) * x).sum(axis=1)
    else:
        raise ValueError("x must be a 1D or 2D array")

