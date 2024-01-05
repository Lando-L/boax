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

"""Quasi Random sampling functions."""

import math

from jax import numpy as jnp
from jax import random, scipy

from boax.core.distributions.normal import Normal
from boax.core.distributions.uniform import Uniform
from boax.core.samplers.functions.utils import primes_less_than
from boax.utils.typing import Array, PRNGKey

# The maximum dimension we support. This is limited by the number of primes in the PRIMES array.
sqrt2 = math.sqrt(2)
MAX_DIMENSION = 10000
PRIMES = primes_less_than(104729 + 1)
assert len(PRIMES) == MAX_DIMENSION


def uniform(base: Array, uniform: Uniform) -> Array:
  return uniform.a + (uniform.b - uniform.a) * base


def normal(base: Array, normal: Normal) -> Array:
  v = 0.5 + (1 - jnp.finfo(base.dtype).eps) * (base - 0.5)
  s = scipy.special.erfinv(2 * v - 1) * sqrt2
  return normal.loc + s * normal.scale


def halton_sequence(key: PRNGKey, num_samples: int, ndims: int) -> Array:
  shuffle_key, correction_key = random.split(key)

  radixes = PRIMES[0:ndims][..., jnp.newaxis]
  indices = jnp.reshape(jnp.arange(num_samples) + 1, (-1, 1, 1))

  max_sizes_by_axes = jnp.floor(jnp.log(num_samples) / jnp.log(radixes) + 1)
  max_size = jnp.max(max_sizes_by_axes)

  exponents_by_axes = jnp.tile(
    jnp.array([jnp.arange(max_size)]), jnp.array([ndims, 1])
  )
  weight_mask = exponents_by_axes < max_sizes_by_axes
  capped_exponents = jnp.where(weight_mask, exponents_by_axes, jnp.zeros(()))
  weights = radixes**capped_exponents

  coeffs = jnp.floor_divide(indices, weights) * weight_mask % radixes
  shuffled = halton_shuffle(shuffle_key, coeffs, radixes) * weight_mask

  base_values = jnp.sum(shuffled / (radixes * weights), axis=-1)
  zero_correction = random.uniform(correction_key, (ndims, 1))

  return (
    base_values + (zero_correction / (radixes**max_sizes_by_axes)).flatten()
  )


def halton_shuffle(key: PRNGKey, coeffs: Array, radixes: Array) -> Array:
  icoeffs = jnp.astype(coeffs, jnp.int32)
  iradixes = jnp.astype(radixes, jnp.int32)
  num_coeffs = coeffs.shape[-1]

  permutations = halton_permutations(key, radixes.flatten(), num_coeffs)
  radix_offsets = jnp.reshape(
    jnp.hstack([jnp.array(0), jnp.cumsum(iradixes[:-1])]), (-1, 1)
  )
  offsets = radix_offsets + jnp.arange(num_coeffs) * jnp.sum(iradixes)

  return permutations[icoeffs + offsets]


def halton_permutations(key: PRNGKey, dims: Array, num_results: int) -> Array:
  max_size = jnp.max(dims)
  mask = jnp.arange(max_size) >= dims[..., jnp.newaxis]

  indices = jnp.stack(jnp.where(~jnp.tile(mask, (num_results, 1, 1))), axis=-1)

  samples = jnp.argsort(
    jnp.where(
      mask,
      jnp.arange(max_size) + 10.0,
      random.uniform(key, (num_results, dims.size, max_size)),
    )
  )

  return samples[*indices.T]
