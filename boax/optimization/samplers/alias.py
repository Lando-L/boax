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

"""Alias for sampling functions."""

from functools import partial

from jax import lax
from jax import numpy as jnp

from boax.optimization.samplers import functions
from boax.optimization.samplers.base import Sampler
from boax.typing import Array
from boax.util import compose


def halton_uniform(
  minval: Array = jnp.zeros((1,)), maxval: Array = jnp.ones((1,))
) -> Sampler:
  out_shape = lax.broadcast_shapes(minval.shape, maxval.shape)

  if out_shape[0] < 1 or out_shape[0] > functions.quasi_random.MAX_DIMENSION:
    raise ValueError(
      'Dimension must be between 1 and {}. Supplied {}'.format(
        functions.quasi_random.MAX_DIMENSION, out_shape[0]
      )
    )

  return compose(
    partial(functions.quasi_random.uniform, minval=minval, maxval=maxval),
    partial(functions.quasi_random.halton_sequence, ndims=out_shape[0]),
  )


def halton_normal(
  loc: Array = jnp.zeros((1,)), scale: Array = jnp.ones((1,))
) -> Sampler:
  out_shape = lax.broadcast_shapes(loc.shape, scale.shape)

  if out_shape[0] < 1 or out_shape[0] > functions.quasi_random.MAX_DIMENSION:
    raise ValueError(
      'Dimension must be between 1 and {}. Supplied {}'.format(
        functions.quasi_random.MAX_DIMENSION, out_shape[0]
      )
    )

  return compose(
    partial(functions.quasi_random.normal, loc=loc, scale=scale),
    partial(functions.quasi_random.halton_sequence, ndims=out_shape[0]),
  )
