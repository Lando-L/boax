# Copyright 2023 The Bojax Authors.
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

"""Alias for search space sampling functions."""

from jax import numpy as jnp

from bojax._src.optimization.spaces import util as space_util
from bojax._src.optimization.spaces.base import SearchSpace
from bojax._src.typing import Array

# The maximum dimension we support. This is limited by the number of primes in the PRIMES array.
MAX_DIMENSION = 10000
PRIMES = space_util.primes_less_than(104729 + 1)
assert len(PRIMES) == MAX_DIMENSION


def continuous(bounds: Array) -> SearchSpace:
  """
  Continuous search space function.

  Samples candidates from a bounded search space based on halton sequences.

  Args:
    bounds: The bounds for the search space.

  Returns:
    The continuous `SearchSpace`.
  """

  ndims = bounds.shape[0]
  lower_bounds = bounds[..., 0]
  upper_bounds = bounds[..., 1]

  if ndims < 1 or ndims > MAX_DIMENSION:
    raise ValueError(
      f'Dimension must be between 1 and {MAX_DIMENSION}. Supplied {ndims}'
    )

  def halton_sequence(num_samples: int) -> Array:
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

  def sample(num_samples: int, **kwargs) -> Array:
    return lower_bounds + (upper_bounds - lower_bounds) * halton_sequence(
      num_samples
    )

  return SearchSpace(
    ndims,
    bounds,
    sample
  )
