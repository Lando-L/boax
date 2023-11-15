from jax import lax, nn
from jax import numpy as jnp

from bojax._src.prediction.bijectors.base import Bijector
from bojax._src.typing import Array, Numeric
from bojax._src.util import identity as identity_fn


def identity() -> Bijector:
    """Creates an identity Bijector."""
    return Bijector(
        identity_fn,
        identity_fn,
    )


def shift(shift: Numeric) -> Bijector:
    """Creates a shift Bijector."""

    def forward(value: Array) -> Array:
        return value + shift

    def inverse(value: Array) -> Array:
        return value - shift

    return Bijector(
        forward,
        inverse,
    )


def scalar_affine(shift: Numeric, scale: Numeric) -> Bijector:
    """Creates an affine Bijector."""

    inv_scale = 1 / scale
    batch_shape = lax.broadcast_shapes(shift.shape, scale.shape)

    def forward(value: Array) -> Array:
        out_shape = lax.broadcast_shapes(batch_shape, value.shape)
        return jnp.broadcast_to(scale, out_shape) * value + jnp.broadcast_to(
            shift, out_shape
        )

    def inverse(value: Array) -> Array:
        out_shape = lax.broadcast_shapes(batch_shape, value.shape)
        return jnp.broadcast_to(inv_scale, out_shape) * (
            value - jnp.broadcast_to(shift, out_shape)
        )

    return Bijector(
        forward,
        inverse,
    )


def softplus():
    """Creates a softplus Bijector."""

    def forward(value: Array) -> Array:
        return nn.softplus(value)

    def inverse(value: Array) -> Array:
        return value + jnp.log(-jnp.expm1(-value))

    return Bijector(
        forward,
        inverse,
    )
