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

"""The normal distribution."""

import math
from typing import NamedTuple

from jax import numpy as jnp
from jax import scipy

from boax.utils.typing import Array, Numeric

sqrt2 = math.sqrt(2)


class Normal(NamedTuple):
  """
  A tuple describing the two parameters of the normal distribution.

  Attributes:
    loc: The location parameter.
    scale: The scale parameter.
  """

  loc: Array
  scale: Array


def normal(loc: Array = jnp.zeros(()), scale: Array = jnp.ones(())) -> Normal:
  """
  Smart constructor for the normal distribution.

  Args:
    loc: The location parameter.
    scale: The scale parameter.

  Returns:
    The `Normal` distribution object.
  """

  return Normal(loc, scale)


def sample(normal: Normal, base_samples: Array) -> Array:
  """
  Samples a normal distribution based on base samples.

  Args:
    normal: The normal distribution.
    base_samples: Normal distributed `*sample_shape x *batch_shape x *event_shape`-dim base samples.

  Returns:
    The samples from the normal distribution.
  """

  transposed_base_samples = jnp.transpose(
    base_samples,
    tuple(range(1, base_samples.ndim)) + (0,),
  )

  samples = (
    transposed_base_samples * normal.scale[..., jnp.newaxis]
    + normal.loc[..., jnp.newaxis]
  )

  transposed_samples = jnp.transpose(
    samples, tuple(range(-1, base_samples.ndim - 1))
  )

  return transposed_samples


def scale(normal: Normal, loc: Numeric, scale: Numeric) -> Normal:
  """
  Scales a normal distribution.

  Args:
    normal: The normal distribution.
    loc: The location parameter.
    scale: The scale parameter

  Returns:
    The scaled normal distribution.
  """

  return Normal(normal.loc * scale + loc, normal.scale)


def pdf(normal: Normal, x: Array) -> Array:
  """
  Probability density function.

  Args:
    normal: The normal distribution.
    values: The values to evaluate.

  Returns:
    The probability density function values.
  """

  return scipy.stats.norm.pdf(x, normal.loc, normal.scale)


def cdf(normal: Normal, x: Array) -> Array:
  """
  Cumulative distribution function.

  Args:
    normal: The normal distribution.
    values: The values to evaluate.

  Returns:
    The cumulative distribution function values.
  """
  return scipy.stats.norm.cdf(x, normal.loc, normal.scale)


def sf(normal: Normal, x: Array) -> Array:
  """
  Survival function.

  Args:
    normal: The normal distribution.
    values: The values to evaluate.

  Returns:
    The survival function values.
  """

  return scipy.stats.norm.sf(x, normal.loc, normal.scale)


def logpdf(normal: Normal, x: Array) -> Array:
  """
  Log probability density function.

  Args:
    mvn: The normal distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.norm.logpdf(x, normal.loc, normal.scale)


def logcdf(normal: Normal, x: Array) -> Array:
  """
  Log cumulative distribution function.

  Args:
    normal: The normal distribution.
    values: The values to evaluate.

  Returns:
    The log cumulative distribution function values.
  """

  return scipy.stats.norm.logcdf(x, normal.loc, normal.scale)


def logsf(normal: Normal, x: Array) -> Array:
  """
  Log survival function.

  Args:
    normal: The normal distribution.
    values: The values to evaluate.

  Returns:
    The log survival function values.
  """

  return scipy.stats.norm.logsf(x, normal.loc, normal.scale)
