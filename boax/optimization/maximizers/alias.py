# Copyright 2023 The Boax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alias for acquisition function maximizers."""

from operator import itemgetter

from jax import numpy as jnp
from jax.scipy.optimize import minimize

from boax.optimization.acquisitions.base import Acquisition
from boax.optimization.maximizers.base import Maximizer
from boax.optimization.maximizers.util import primes_less_than
from boax.typing import Array, Numeric
from boax.util import compose

# The maximum dimension we support. This is limited by the number of primes in the PRIMES array.
MAX_DIMENSION = 10000
PRIMES = primes_less_than(104729 + 1)
assert len(PRIMES) == MAX_DIMENSION


def halton_sequence(num_samples: int, ndims: int) -> Array:
  if ndims < 1 or ndims > MAX_DIMENSION:
    raise ValueError(
      f'Dimension must be between 1 and {MAX_DIMENSION}. Supplied {ndims}'
    )
  
  radixes = PRIMES[0:ndims][..., jnp.newaxis]
  indices = jnp.reshape(jnp.arange(num_samples) + 1, [-1, 1, 1])

  max_sizes_by_axes = jnp.floor(jnp.log(num_samples) / jnp.log(radixes) + 1)
  max_size = jnp.max(max_sizes_by_axes)

  exponents_by_axes = jnp.tile(
    jnp.array([jnp.arange(max_size)]), jnp.array([ndims, 1])
  )
  weight_mask = exponents_by_axes < max_sizes_by_axes
  capped_exponents = jnp.where(weight_mask, exponents_by_axes, jnp.zeros(()))
  weights = radixes**capped_exponents

  coeffs = jnp.floor_divide(indices, weights) * weight_mask % radixes

  return jnp.sum(coeffs / (radixes * weights), axis=-1)


def bfgs(num_initial_samples: int, bounds: Array) -> Maximizer:
  """
  The BFGS acquisition function maximizer.

  Args:
    num_initial_samples: The number of initial samples used by the maximizer.
    bounds: The bounds of the search space.

  Returns:
    The corresponding `Maximizer`.
  """

  ndims = bounds.shape[0]
  lower_bounds = bounds[..., 0]
  upper_bounds = bounds[..., 1]

  initial_samples = lower_bounds + (upper_bounds - lower_bounds) * halton_sequence(
    num_initial_samples,
    ndims
  )

  def maximizer(acquisition: Acquisition) -> Numeric:
    results = minimize(
      fun=compose(
        jnp.negative, jnp.sum, acquisition, itemgetter((..., jnp.newaxis))
      ),
      x0=initial_samples[..., 0],
      method='bfgs',
    )

    candidates = jnp.clip(
      results.x[..., jnp.newaxis],
      a_min=bounds[..., 0],
      a_max=bounds[..., 1],
    )

    return candidates, acquisition(candidates)

  return maximizer
