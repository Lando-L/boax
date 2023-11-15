from typing import Protocol

from bojax._src.typing import Array


class SearchSpace(Protocol):
    def __call__(self, num_samples: int, **kwargs) -> Array:
        """Samples `num_samples` points from the search space."""
