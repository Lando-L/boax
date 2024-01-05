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


def beta(a: Array = jnp.ones(()), b: Array = jnp.ones(())) -> Array:
  return Beta(a, b)


def pdf(values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))) -> Array:
  return scipy.stats.beta.pdf(values, beta.a, beta.b)


def cdf(values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))) -> Array:
  return scipy.stats.norm.cdf(values, beta.a, beta.b)


def sf(values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))) -> Array:
  return scipy.stats.norm.sf(values, beta.a, beta.b)


def logpdf(
  values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))
) -> Array:
  return scipy.stats.norm.logpdf(values, beta.a, beta.b)


def logcdf(
  values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))
) -> Array:
  return scipy.stats.norm.logcdf(values, beta.a, beta.b)


def logsf(
  values: Array, beta: Beta = Beta(jnp.ones(()), jnp.ones(()))
) -> Array:
  return scipy.stats.norm.logsf(values, beta.a, beta.b)
