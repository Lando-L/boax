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

import math

from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.distributions.normal import Normal
from boax.core.distributions.uniform import Uniform
from boax.core.samplers import functions
from boax.core.samplers.base import Sampler
from boax.utils.typing import Array, PRNGKey, Shape


def uniform(key: PRNGKey, sample_shape: Shape) -> Sampler[Uniform]:
  """
  The i.i.d. uniform sampler.

  Example:
    >>> sampler = uniform(key, (128,))
    >>> samples = sampler(Uniform(0, 1))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(uniform: Uniform) -> Array:
    batch_shape = uniform.a.shape[:-1]
    event_shape = jnp.broadcast_shapes(uniform.a.shape[-1], uniform.b.shape[-1])
    base_samples = random.uniform(key, sample_shape + batch_shape + event_shape)

    return distributions.uniform.sample(uniform, base_samples)

  return sample


def normal(key: PRNGKey, sample_shape: Shape) -> Sampler[Normal]:
  """
  The i.i.d. normal sampler.

  Example:
    >>> sampler = normal(key, (128,))
    >>> samples = sampler(Normal(0, 1))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(normal: Normal) -> Array:
    batch_shape = normal.loc.shape[:-1]
    event_shape = jnp.broadcast_shapes(
      normal.loc.shape[-1], normal.scale.shape[-1]
    )
    base_samples = random.normal(key, sample_shape + batch_shape + event_shape)

    return distributions.normal.sample(normal, base_samples)

  return sample


def multivariate_normal(
  key: PRNGKey, sample_shape: Shape
) -> Sampler[MultivariateNormal]:
  """
  The i.i.d. multivariate normal sampler.

  Example:
    >>> sampler = multivariate_normal(key, (128,))
    >>> samples = sampler(MultivariateNormal(jnp.zeros((1,)), jnp.identity(1)))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(mvn: MultivariateNormal) -> Array:
    batch_shape = mvn.mean.shape[:-1]
    event_shape = jnp.broadcast_shapes(mvn.mean.shape[-1], mvn.cov.shape[-2])
    base_samples = random.normal(key, sample_shape + batch_shape + event_shape)

    return distributions.multivariate_normal.sample(mvn, base_samples)

  return sample


def halton_uniform(key: PRNGKey, sample_shape: Shape) -> Sampler[Uniform]:
  """
  The quasi-MC uniform sampler based on halton sequences.

  Example:
    >>> sampler = halton_uniform(key, (128,))
    >>> samples = sampler(Uniform(0, 1))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(uniform: Uniform) -> Array:
    batch_shape = uniform.a.shape[:-1]
    event_shape = jnp.broadcast_shapes(uniform.a.shape[-1], uniform.b.shape[-1])
    out_shape = batch_shape + event_shape
    ndims = math.prod(out_shape)

    if ndims < 1 or ndims > functions.quasi_random.MAX_DIMENSION:
      raise ValueError(
        'Dimension must be between 1 and {}. Supplied {}'.format(
          functions.quasi_random.MAX_DIMENSION, ndims
        )
      )

    base_samples = functions.quasi_random.halton_sequence(
      key, sample_shape, ndims
    )
    reshaped = jnp.reshape(base_samples, sample_shape + out_shape)

    return distributions.uniform.sample(uniform, reshaped)

  return sample


def halton_normal(key: PRNGKey, sample_shape: Shape) -> Sampler[Normal]:
  """
  The quasi-MC normal sampler based on halton sequences.

  Example:
    >>> sampler = halton_normal(key, (128,))
    >>> samples = sampler(Normal(0, 1))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(normal: Normal) -> Array:
    out_shape = jnp.broadcast_shapes(normal.loc.shape, normal.scale.shape)
    ndims = math.prod(out_shape)

    if ndims < 1 or ndims > functions.quasi_random.MAX_DIMENSION:
      raise ValueError(
        'Dimension must be between 1 and {}. Supplied {}'.format(
          functions.quasi_random.MAX_DIMENSION, ndims
        )
      )

    base_samples = functions.quasi_random.halton_sequence(
      key, sample_shape, ndims
    )
    normal_samples = functions.utils.ratio_of_uniforms(base_samples)
    reshaped = jnp.reshape(normal_samples, sample_shape + out_shape)

    return distributions.normal.sample(normal, reshaped)

  return sample


def halton_multivariate_normal(
  key: PRNGKey, sample_shape: Shape
) -> Sampler[MultivariateNormal]:
  """
  The quasi-MC multivariate normal sampler based on halton sequences.

  Example:
    >>> sampler = halton_multivariate_normal(key, (128,))
    >>> samples = sampler(MultivariateNormal(jnp.zeros((1,)), jnp.identity(1)))

  Args:
    key: The pseudo-random number genererator key.
    sample_shape: The sample shape.

  Returns:
    The corresponding `Sampler`.
  """

  def sample(mvn: MultivariateNormal) -> Array:
    batch_shape = mvn.mean.shape[:-1]
    event_shape = jnp.broadcast_shapes(mvn.mean.shape[-1], mvn.cov.shape[-2])
    out_shape = batch_shape + event_shape
    ndims = math.prod(out_shape)

    if ndims < 1 or ndims > functions.quasi_random.MAX_DIMENSION:
      raise ValueError(
        'Dimension must be between 1 and {}. Supplied {}'.format(
          functions.quasi_random.MAX_DIMENSION, ndims
        )
      )

    base_samples = functions.quasi_random.halton_sequence(
      key, sample_shape, ndims
    )
    normal_samples = functions.utils.ratio_of_uniforms(base_samples)
    reshaped = jnp.reshape(normal_samples, sample_shape + out_shape)

    return distributions.multivariate_normal.sample(mvn, reshaped)

  return sample
