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

"""The uniform distribution."""

from typing import NamedTuple

from jax import numpy as jnp
from jax import scipy

from boax.utils.typing import Array


class Uniform(NamedTuple):
  """
  A tuple describing the two parameters of the uniform distribution.

  Attributes:
    a: The minium value parameter.
    b: The maximum value parameter.
  """

  a: Array
  b: Array


def uniform(a: Array = jnp.zeros(()), b: Array = jnp.ones(())) -> Uniform:
  """
  Smart constructor for the uniform distribution.

  Args:
    a: The minium value parameter.
    b: The maximum value parameter.

  Returns:
    The `Uniform` distribution object.
  """

  return Uniform(a, b)


def sample(uniform: Uniform, base_samples: Array) -> Array:
  """
  Samples a uniform distribution based on base samples.

  Args:
    uniform: The uniform distribution.
    base_samples: The uniform distributed base samples.

  Returns:
    The samples from the uniform distribution.
  """

  return uniform.a + (uniform.b - uniform.a) * base_samples


def pdf(uniform: Uniform, values: Array) -> Array:
  """
  Probability density function.

  Args:
    uniform: The uniform distribution.
    values: The values to evaluate.

  Returns:
    The probability density function values.
  """

  return scipy.stats.uniform.pdf(values, uniform.a, uniform.b - uniform.a)


def logpdf(uniform: Uniform, values: Array) -> Array:
  """
  Log probability density function.

  Args:
    uniform: The uniform distribution.
    values: The values to evaluate.

  Returns:
    The log probability density function values.
  """

  return scipy.stats.uniform.logpdf(values, uniform.a, uniform.b - uniform.a)
