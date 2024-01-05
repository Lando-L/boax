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

from boax.core.distributions.normal import Normal
from boax.core.distributions.uniform import Uniform
from boax.core.samplers import functions
from boax.core.samplers.base import Sampler
from boax.utils.functools import compose


def halton_uniform(
  uniform: Uniform = Uniform(jnp.zeros((1,)), jnp.ones((1,))),
) -> Sampler:
  """
  The quasi-MC uniform sampler based on halton sequences.

  Example:
    >>> sampler = halton_uniform(uniform)
    >>> base_samples = sampler(key, 128)

  Args:
    uniform: The base uniform distribution.

  Returns:
    The corresponding `Sampler`.
  """

  out_shape = lax.broadcast_shapes(uniform.a.shape, uniform.b.shape)

  if out_shape[0] < 1 or out_shape[0] > functions.quasi_random.MAX_DIMENSION:
    raise ValueError(
      'Dimension must be between 1 and {}. Supplied {}'.format(
        functions.quasi_random.MAX_DIMENSION, out_shape[0]
      )
    )

  return compose(
    partial(functions.quasi_random.uniform, uniform=uniform),
    partial(functions.quasi_random.halton_sequence, ndims=out_shape[0]),
  )


def halton_normal(
  normal: Normal = Normal(jnp.zeros((1,)), jnp.ones((1,))),
) -> Sampler:
  """
  The quasi-MC normal sampler based on halton sequences.

  Example:
    >>> sampler = halton_normal(normal)
    >>> base_samples = sampler(key, 128)

  Args:
    normal: The base normal distribution.

  Returns:
    The corresponding `Sampler`.
  """

  out_shape = lax.broadcast_shapes(normal.loc.shape, normal.scale.shape)

  if out_shape[0] < 1 or out_shape[0] > functions.quasi_random.MAX_DIMENSION:
    raise ValueError(
      'Dimension must be between 1 and {}. Supplied {}'.format(
        functions.quasi_random.MAX_DIMENSION, out_shape[0]
      )
    )

  return compose(
    partial(functions.quasi_random.normal, normal=normal),
    partial(functions.quasi_random.halton_sequence, ndims=out_shape[0]),
  )
