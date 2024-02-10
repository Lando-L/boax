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

"""The multivariate normal distribution."""

from typing import NamedTuple

from jax import numpy as jnp
from jax import scipy

from boax.core.distributions.normal import Normal
from boax.utils.typing import Array


class MultivariateNormal(NamedTuple):
  """
  A tuple describing the two parameters of the multivariate normal distribution.

  Attributes:
    mean: The mean vector parameter.
    cov: The covariance matrix parameter.
  """

  mean: Array
  cov: Array


def multivariate_normal(
  mean: Array = jnp.zeros((1,)), cov: Array = jnp.identity(1)
) -> MultivariateNormal:
  """
  Smart constructor for the multivariate normal distribution.

  Args:
    mean: The mean vector parameter.
    cov: The covariance matrix parameter.

  Returns:
    The `MultivariateNormal` distribution object.
  """

  return MultivariateNormal(mean, cov)


def as_normal(mvn: MultivariateNormal) -> Normal:
  """
  Transforms a multivariate normal distribution
  into a batched normal distribution.

  Args:
    mvn: The multivariate normal distribution.

  Returns:
    The batched `Normal` distribution object.
  """

  return Normal(mvn.mean, jnp.sqrt(jnp.diag(mvn.cov)))


def sample(mvn: MultivariateNormal, base_samples: Array) -> Array:
  """
  Samples a multivariate normal distribution based on base samples.

  Args:
    mvn: The multivariate normal distribution.
    base_samples: The normal distributed base samples.

  Returns:
    The samples from the multivariate normal distribution.
  """

  return mvn.mean + jnp.dot(
    scipy.linalg.cholesky(mvn.cov, lower=True), base_samples
  )


def pdf(mvn: MultivariateNormal, values: Array) -> Array:
  """
  Probability density function.

  Args:
    mvn: The multivariate normal distribution.
    values: The values to evaluate.

  Returns:
    The probability density function values.
  """

  return scipy.stats.multivariate_normal.pdf(values, mvn.mean, mvn.cov)


def logpdf(mvn: MultivariateNormal, values: Array) -> Array:
  """
  Log probability density function.

  Args:
    mvn: The multivariate normal distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.multivariate_normal.logpdf(values, mvn.mean, mvn.cov)
