from typing import NamedTuple, Protocol

from bojax._src.typing import Array


class BijectorForwardFn(Protocol):
    def __call__(self, value: Array) -> Array:
        """Computes y = f(x)"""


class BijectorInverseFn(Protocol):
    def __call__(self, value: Array) -> Array:
        """Computes x = f^{-1}(y)"""


class Bijector(NamedTuple):
    forward: BijectorForwardFn
    inverse: BijectorInverseFn
