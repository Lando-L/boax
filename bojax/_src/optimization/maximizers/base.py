from typing import Protocol, Tuple

from bojax._src.optimization.acquisitions.base import Acquisition
from bojax._src.typing import Array


class Maximizer(Protocol):
    def __call__(self, acquisition: Acquisition, bounds: Array) -> Tuple[Array, Array]:
        """Maximizes an acquisition within the given bounds"""
