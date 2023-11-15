from jax import numpy as jnp

from bojax._src.prediction.kernels.base import Kernel, kernel
from bojax._src.typing import Array, Numeric


def squared_distance(x: Array, y: Array) -> Array:
    return jnp.sum(x**2) + jnp.sum(y**2) - 2 * jnp.inner(x, y)


def rbf(length_scale: Numeric) -> Kernel:
    def kernel_fn(x, y):
        return jnp.exp(-0.5 * squared_distance(x / length_scale, y / length_scale))

    return kernel(kernel_fn)


def matern_half(length_scale: Numeric) -> Kernel:
    def kernel_fn(x, y):
        return jnp.exp(
            -jnp.linalg.norm(jnp.subtract(x / length_scale, y / length_scale))
        )

    return kernel(kernel_fn)


def matern_three_halves(length_scale: Numeric) -> Kernel:
    def kernel_fn(x, y):
        K = jnp.linalg.norm(
            jnp.subtract(x / length_scale, y / length_scale)
        ) * jnp.sqrt(3)
        K = (1.0 + K) * jnp.exp(-K)
        return K

    return kernel(kernel_fn)


def matern_five_halves(length_scale: Numeric) -> Kernel:
    def kernel_fn(x, y):
        K = jnp.linalg.norm(
            jnp.subtract(x / length_scale, y / length_scale)
        ) * jnp.sqrt(5)
        K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
        return K

    return kernel(kernel_fn)


def periodic(length_scale: Numeric, variance: Numeric, period: Numeric) -> Kernel:
    def kernel_fn(x, y):
        sine_squared = (jnp.sin(jnp.pi * (x - y) / period) / length_scale) ** 2
        K = variance * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K
    
    return kernel(kernel_fn)
