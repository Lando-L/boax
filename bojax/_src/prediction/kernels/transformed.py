from typing import Callable

from jax import numpy as jnp

from bojax._src.prediction.kernels.base import Kernel
from bojax._src.typing import Numeric


def scale(amplitude: Numeric, inner: Kernel) -> Kernel:
    def kernel(x, y):
        return amplitude * inner(x, y)

    return kernel


def combine(operator: Callable, *kernels: Kernel) -> Kernel:
    def kernel(x, y):
        return operator(jnp.stack([k(x, y) for k in kernels]))
    
    return kernel
