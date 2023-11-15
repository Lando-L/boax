from typing import Protocol

from bojax._src.typing import Array


class Mean(Protocol):
    def __call__(self, value: Array) -> Array:
        """Returns the mean."""
