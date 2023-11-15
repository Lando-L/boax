from operator import itemgetter

from jax import numpy as jnp
from jax.scipy.optimize import minimize

from bojax._src.optimization.acquisitions.base import Acquisition
from bojax._src.optimization.maximizers.base import Maximizer
from bojax._src.optimization.space.alias import continuous
from bojax._src.typing import Array, Numeric
from bojax._src.util import compose


def bfgs(num_initial_samples: int) -> Maximizer:
    def maximizer(acquisition: Acquisition, bounds: Array) -> Numeric:        
        results = minimize(
            fun=compose(jnp.negative, jnp.sum, acquisition, itemgetter((..., jnp.newaxis))),
            x0=continuous(bounds)(num_initial_samples)[..., 0],
            method='bfgs'
        )

        candidates = jnp.clip(results.x[..., jnp.newaxis], a_min=bounds[..., 0], a_max=bounds[..., 1])

        return candidates, acquisition(candidates)

    return maximizer
