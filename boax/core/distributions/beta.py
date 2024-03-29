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

"""The beta distribution."""

from typing import NamedTuple

from jax import numpy as jnp
from jax import scipy

from boax.utils.typing import Array


class Beta(NamedTuple):
  """
  A tuple describing the two shape parameters of the beta distribution.

  Attributes:
    a: The alpha parameter.
    b: The beta parameter.
  """

  a: Array
  b: Array


def beta(a: Array = jnp.ones(()), b: Array = jnp.ones(())) -> Beta:
  """
  Smart constructor for the beta distribution.

  Args:
    a: The alpha parameter.
    b: The beta parameter.

  Returns:
    The `Beta` distribution object.
  """

  return Beta(a, b)


def pdf(beta: Beta, values: Array) -> Array:
  """
  Probability density function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The probability density function values.
  """

  return scipy.stats.beta.pdf(values, beta.a, beta.b)


def cdf(beta: Beta, values: Array) -> Array:
  """
  Cumulative distribution function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The cumulative distribution function values.
  """

  return scipy.stats.beta.cdf(values, beta.a, beta.b)


def sf(beta: Beta, values: Array) -> Array:
  """
  Survival function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The survival function values.
  """

  return scipy.stats.beta.sf(values, beta.a, beta.b)


def logpdf(beta: Beta, values: Array) -> Array:
  """
  Log probability density function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.beta.logpdf(values, beta.a, beta.b)


def logcdf(beta: Beta, values: Array) -> Array:
  """
  Log cumulative distribution function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The log cumulative distribution function values.
  """

  return scipy.stats.beta.logcdf(values, beta.a, beta.b)


def logsf(beta: Beta, values: Array) -> Array:
  """
  Log survival function.

  Args:
    beta: The beta distribution.
    values: The values to evaluate.

  Returns:
    The log survival function values.
  """

  return scipy.stats.beta.logsf(values, beta.a, beta.b)
