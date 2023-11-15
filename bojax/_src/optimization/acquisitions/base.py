from typing import Protocol

from bojax._src.typing import Array


class Acquisition(Protocol):
    def __call__(self, value: Array, **kwargs) -> Array:
        """Evaluates the acquisition function on the candidate set X."""
