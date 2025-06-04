from flax import struct
import jax
from typing import Callable, Tuple


@struct.dataclass
class BaseDomain:
    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def project(self, array: jax.Array) -> jax.Array:
        """Projects the given input to the domain. Returns jnp.nan for out of bounds."""
        raise NotImplementedError

    def maximize(
        self, f: Callable[[jax.Array], jax.Array]
    ) -> Tuple[jax.Array, jax.Array]:
        """Returns the argmax and max of f over the domain."""
        raise NotImplementedError
