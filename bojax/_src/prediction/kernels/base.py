from typing import Protocol

from jax import vmap

from bojax._src.typing import Array


class Kernel(Protocol):
    def __call__(self, x: Array, y: Array) -> Array:
        """Calculates kernel function to pairs of inputs."""


def kernel(kernel_fn: Kernel) -> Kernel:
    return vmap(vmap(kernel_fn, in_axes=(None, 0)), in_axes=(0, None))
