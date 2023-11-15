from jax import numpy as jnp
from jax import vmap

from bojax._src.prediction.means.base import Mean
from bojax._src.typing import Array, Numeric
from bojax._src.util import const


def zero() -> Mean:
    return vmap(const(jnp.zeros(())))


def constant(x: Numeric) -> Mean:
    return vmap(const(x))


def linear(scale: Array, bias: Numeric) -> Mean:
    def mean(value: Array) -> Array:
        return jnp.dot(scale, value) + bias

    return vmap(mean)
