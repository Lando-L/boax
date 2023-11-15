from operator import attrgetter

from bojax._src.prediction.bijectors.base import Bijector
from bojax._src.util import compose


def chain(*bijectors: Bijector) -> Bijector:
    """Chains a sequence of Bijectors."""

    return Bijector(
        compose(*map(attrgetter("forward"), bijectors)),
        compose(*map(attrgetter("inverse"), reversed(bijectors))),
    )
