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

"""Utils for search space sampling functions."""

import numpy as np
from jax import numpy as jnp

from boax.typing import Array


def primes_less_than(n):
  """
  Sorted array of primes such that `2 <= prime < n`.

  Args:
    n: The upper bound for primes.

  Returns:
    The sorted array of primes.
  """

  j = 3
  primes = np.ones((n + 1) // 2, dtype=bool)

  while j * j <= n:
    if primes[j // 2]:
      primes[j * j // 2 :: j] = False
    j += 2

  ret = 2 * np.where(primes)[0] + 1
  ret[0] = 2  # :(

  return ret


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
