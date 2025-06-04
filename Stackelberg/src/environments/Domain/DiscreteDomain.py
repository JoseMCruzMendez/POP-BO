import jax
from flax import struct
from jax import numpy as jnp
from typing import Callable, Tuple, Union

from src.environments.Domain.BaseDomain import BaseDomain


@struct.dataclass
class DiscreteDomain(BaseDomain):
    """
    A discrete domain with a given number of elements and optional features.
    Use .create() to create a DiscreteDomain with features.
    """

    num_elements: int
    indices: jax.Array = None
    features: jax.Array = None

    @property
    def has_features(self):
        return self.features is not None

    @property
    def feature_dim(self):
        if self.has_features:
            return self.features.shape[-1]
        else:
            raise ValueError("No features available. feature_dim is not defined.")

    @property
    def feature_matrix(self):
        if self.has_features:
            return self.features[:-1]
        else:
            raise ValueError("No features available. feature_matrix is not defined.")

    @classmethod
    def create(cls, num_elements: int, features: jax.Array = None):
        """Creates a DiscreteDomain with the given number of elements and features."""
        if features is not None:
            if len(features.shape) != 2:
                raise ValueError(
                    f"Features shape {features.shape} must be 2-dimensional with axis: (num_elements, dim)."
                )
            if features.shape[0] != num_elements:
                raise ValueError(
                    f"Features shape's first entry {features.shape} does not match num_elements {num_elements}."
                )
            features = jnp.concatenate(
                [features, jnp.full((1, features.shape[1]), jnp.nan)], axis=0
            )
        return cls(
            num_elements=num_elements,
            indices=jnp.arange(num_elements, dtype=jnp.int32),
            features=features,
        )

    def project(self, array: Union[int, jax.Array]) -> jax.Array:
        """Projects the given input to the domain. Returns -1 for out of bounds."""
        return jnp.where(
            jnp.isin(array, self.indices),
            array,
            -1,
        )

    def get_feature(
            self,
            index: Union[int, jax.Array],
            feature_encoding: str = "one_hot",
    ) -> jax.Array:
        """
        Returns the feature of the given index. If the index is nan, returns the feature of the last index.
        :param index: The index of the feature. Shape: (batch_size,).
        :param feature_encoding: The encoding of the feature if self.feature is nont. Currently only supports "one_hot".
        :return: The feature of the given index. Shape: (batch_size, feature_dim).
        """
        index_nan_mask = jnp.isnan(index)
        index = jnp.where(jnp.isnan(index), self.num_elements, index).astype(
            jnp.int32
        )  # Where nan, return last index
        if self.has_features:
            return self.features[index]
        else:
            if feature_encoding == "one_hot":
                features = jax.nn.one_hot(index, self.num_elements)
                features = features.at[index_nan_mask].set(jnp.full((jnp.sum(index_nan_mask), self.num_elements), jnp.nan))
                return features
            else:
                raise ValueError(f"Unknown feature_encoding {feature_encoding}.")

    def maximize(
        self,
        f: Callable[[jax.Array], jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Returns the argmax and max of f over the domain.
        :param f: The function to maximize. Takes in a feature and returns a scalar value.
        :return: The argmax and max of f over the domain.
        """
        values = jax.vmap(f)(self.indices)
        max_idx = jnp.argmax(values)
        max_value = jnp.max(values)
        return max_idx, max_value
