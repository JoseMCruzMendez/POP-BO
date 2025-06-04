import jax.numpy as jnp
import jax
from flax import struct
from typing import Tuple


@struct.dataclass
class Kernel:
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the cross covariance between the given inputs.
        :param x: array of Shape: (n, n_features) (or (n_features,) for a single input)
        :param y: array of Shape: (m, n_features) (or (n_features,) for a single input)
        :return: array of Shape: (n, m)
            n and m are equal to 1 if input shape was not specified.
        """
        raise NotImplementedError

    def gram(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.cross_covariance(x, x)


@struct.dataclass
class LinearKernel(Kernel):
    variance: float = 1.0

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
        return self.variance * jnp.dot(x, y.T)


@struct.dataclass
class RBFKernel(Kernel):
    variance: float = 1.0
    length_scale: float = 1.0

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x2d, y2d = jnp.atleast_2d(x), jnp.atleast_2d(y)
        covar = self.variance * jnp.exp(
            -0.5
            * jnp.sum((x2d[:, None, :] - y2d[None, :, :]) ** 2, axis=-1)
            / self.length_scale**2
        )
        # Remove added dimensions
        if x.ndim == 1 and y.ndim == 1:
            return covar.squeeze()
        elif x.ndim == 1 or y.ndim == 1:
            return covar.reshape(-1)
        else:
            return covar


@struct.dataclass
class Matern12Kernel(Kernel):
    variance: float = 1.0
    length_scale: float = 1.0

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
        return self.variance * jnp.exp(
            -jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
            / self.length_scale
        )


@struct.dataclass
class Matern32Kernel(Kernel):
    variance: float = 1.0
    length_scale: float = 1.0

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
        scaled_distance = (
            jnp.sqrt(3)
            * jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
            / self.length_scale
        )
        return self.variance * (1 + scaled_distance) * jnp.exp(-scaled_distance)


@struct.dataclass
class Matern53Kernel(Kernel):
    variance: float = 1.0
    length_scale: float = 1.0

    def cross_covariance(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
        scaled_distance = (
            jnp.sqrt(5)
            * jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
            / self.length_scale
        )
        return (
            self.variance
            * (1 + scaled_distance + scaled_distance**2 / 3.0)
            * jnp.exp(-scaled_distance)
        )


@struct.dataclass
class DuellingWrapper:
    kernel: Kernel

    def gram(self, x: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        return self.cross_covariance(x, x)

    def cross_covariance(self, x: Tuple[jnp.ndarray, jnp.ndarray], y: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Computes the cross covariance between two duelling arms.
        :param x: Tuple of two arrays of Shape: (n, n_features)
        :param y: Tuple of two arrays of Shape: (m, n_features)
        :return: array of Shape: (n, m)
        """
        x1, x2 = x
        y1, y2 = y
        return (
            self.kernel.cross_covariance(x1, y1)
            + self.kernel.cross_covariance(x2, y2)
            - self.kernel.cross_covariance(x1, y2)
            - self.kernel.cross_covariance(x2, y1)
        )