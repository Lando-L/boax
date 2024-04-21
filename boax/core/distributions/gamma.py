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

"""The gamma distribution."""

from typing import NamedTuple

from jax import numpy as jnp
from jax import scipy

from boax.utils.typing import Array


class Gamma(NamedTuple):
  """
  A tuple describing the shape and rate parameters of the gamma distribution.

  Attributes:
    a: The shape parameter.
    b: The rate parameter.
  """

  a: Array
  b: Array


def gamma(a: Array, b: Array = jnp.ones(())) -> Gamma:
  """
  Smart constructor for the gamma distribution.

  Args:
    a: The shape parameter.
    b: The rate parameter.

  Returns:
    The `Gamma` distribution object.
  """

  return Gamma(a, b)


def pdf(gamma: Gamma, values: Array) -> Array:
  """
  Probability density function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The probability density function values.
  """

  return scipy.stats.gamma.pdf(values, gamma.a, 0, 1 / gamma.b)


def cdf(gamma: Gamma, values: Array) -> Array:
  """
  Cumulative distribution function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The cumulative distribution function values.
  """

  return scipy.stats.gamma.cdf(values, gamma.a, 0, 1 / gamma.b)


def sf(gamma: Gamma, values: Array) -> Array:
  """
  Survival function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The survival function values.
  """

  return scipy.stats.gamma.sf(values, gamma.a, 0, 1 / gamma.b)


def logpdf(gamma: Gamma, values: Array) -> Array:
  """
  Log probability density function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.gamma.logpdf(values, gamma.a, 0, 1 / gamma.b)


def logcdf(gamma: Gamma, values: Array) -> Array:
  """
  Log cumulative distribution function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The log cumulative distribution function values.
  """

  return scipy.stats.gamma.logcdf(values, gamma.a, 0, 1 / gamma.b)


def logsf(gamma: Gamma, values: Array) -> Array:
  """
  Log survival function.

  Args:
    gamma: The gamma distribution.
    values: The values to evaluate.

  Returns:
    The log survival function values.
  """

  return scipy.stats.gamma.logsf(values, gamma.a, 0, 1 / gamma.b)
