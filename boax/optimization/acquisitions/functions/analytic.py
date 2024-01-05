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

"""Analytic acquisition functions."""

import math

from jax import numpy as jnp
from jax import scipy

from boax.core import distributions
from boax.core.distributions.normal import Normal
from boax.utils.typing import Array, Numeric

log2 = math.log(2)
inv_sqrt2 = 1 / math.sqrt(2)
c1 = math.log(2 * math.pi) / 2
c2 = math.log(math.pi / 2) / 2


def poi(normal: Normal, best: Numeric) -> Array:
  u = (normal.loc - best) / normal.scale
  return distributions.normal.cdf(u)


def lpoi(normal: Normal, best: Numeric) -> Array:
  u = (normal.loc - best) / normal.scale
  return distributions.normal.logcdf(u)


def ei(normal: Normal, best: Numeric) -> Array:
  u = (normal.loc - best) / normal.scale
  return (
    distributions.normal.pdf(u) + u * distributions.normal.cdf(u)
  ) * normal.scale


def lei(normal: Normal, best: Numeric) -> Array:
  def log1mexp(x):
    upper = jnp.where(x > -log2, x, -log2)
    lower = jnp.where(x <= -log2, x, -log2)

    return jnp.where(
      x > -log2, jnp.log(-jnp.expm1(upper)), jnp.log1p(-jnp.exp(lower))
    )

  def logerfcx(x):
    upper = jnp.where(x > 0.0, x, 0.0)
    lower = jnp.where(x <= 0.0, x, 0.0)

    return jnp.where(
      x > 0,
      jnp.log(jnp.exp(upper**2) * scipy.special.erfc(upper)),
      jnp.log(scipy.special.erfc(lower)) + lower**2,
    )

  def log_ei_upper(x):
    return jnp.log(
      distributions.normal.pdf(x) + x * distributions.normal.cdf(x)
    )

  def log_ei_lower(x):
    return (
      -(x**2) / 2 - c1 + log1mexp(logerfcx(-x * inv_sqrt2) * jnp.abs(x) + c2)
    )

  def logh(x):
    upper = jnp.where(x > -1.0, x, -1.0)
    lower = jnp.where(x <= -1.0, x, -1.0)

    return jnp.where(x > -1, log_ei_upper(upper), log_ei_lower(lower))

  u = (normal.loc - best) / normal.scale
  return logh(u) + jnp.log(normal.scale)


def ucb(normal: Normal, beta: Numeric) -> Array:
  return normal.loc + jnp.sqrt(beta) * normal.scale
