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

from boax.utils.typing import Array

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
  return Normal(loc, scale)


def sample(normal: Normal, base_samples: Array) -> Array:
  return normal.loc + base_samples * normal.scale


def pdf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.pdf(x, normal.loc, normal.scale)


def cdf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.cdf(x, normal.loc, normal.scale)


def sf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.sf(x, normal.loc, normal.scale)


def logpdf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.logpdf(x, normal.loc, normal.scale)


def logcdf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.logcdf(x, normal.loc, normal.scale)


def logsf(normal: Normal, x: Array) -> Array:
  return scipy.stats.norm.logsf(x, normal.loc, normal.scale)
