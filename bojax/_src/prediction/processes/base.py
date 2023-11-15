from typing import Generic, Protocol, TypeVar

from bojax._src.typing import Array

T = TypeVar("T")


class Process(Protocol, Generic[T]):
    def __call__(self, value: Array) -> T:
        """Stochastic Process."""
