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

"""The poisson distribution."""

from typing import NamedTuple

from jax import scipy

from boax.utils.typing import Array


class Poisson(NamedTuple):
  """
  A tuple describing the rate parameter of the poisson distribution.

  Attributes:
    mu: The rate parameter.
  """

  mu: Array


def poisson(mu: Array) -> Poisson:
  """
  Smart constructor for the beta distribution.

  Args:
    mu: The rate parameter.

  Returns:
    The `Poisson` distribution object.
  """

  return Poisson(mu)


def pmf(poisson: Poisson, values: Array) -> Array:
  """
  Probability mass function.

  Args:
    poisson: The poisson distribution.
    values: The values to evaluate.

  Returns:
    The probability mass function values.
  """

  return scipy.stats.poisson.pmf(values, poisson.mu)


def cdf(poisson: Poisson, values: Array) -> Array:
  """
  Cumulative distribution function.

  Args:
    poisson: The poisson distribution.
    values: The values to evaluate.

  Returns:
    The cumulative distribution function values.
  """

  return scipy.stats.poisson.cdf(values, poisson.mu)


def logpmf(poisson: Poisson, values: Array) -> Array:
  """
  Log probability density function.

  Args:
    poisson: The poisson distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.poisson.logpdf(values, poisson.mu)
