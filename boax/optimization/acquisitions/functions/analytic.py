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
from typing import Tuple

from jax import numpy as jnp
from jax import scipy

from boax.typing import Array, Numeric

log2 = math.log(2)
inv_sqrt2 = 1 / math.sqrt(2)
c1 = math.log(2 * math.pi) / 2
c2 = math.log(math.pi / 2) / 2


def analytic(loc: Array, cov: Array) -> Tuple[Array, Array]:
  return loc, jnp.sqrt(jnp.diag(cov))


def scaled_improvement(loc: Array, scale: Array, best: Numeric) -> Array:
  return (loc - best) / scale


def ei(loc: Array, scale: Array, best: Numeric) -> Array:
  u = scaled_improvement(loc, scale, best)
  return (scipy.stats.norm.pdf(u) + u * scipy.stats.norm.cdf(u)) * scale


def lei(loc: Array, scale: Array, best: Numeric) -> Array:
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
    return jnp.log(scipy.stats.norm.pdf(x) + x * scipy.stats.norm.cdf(x))

  def log_ei_lower(x):
    return (
      -(x**2) / 2 - c1 + log1mexp(logerfcx(-x * inv_sqrt2) * jnp.abs(x) + c2)
    )

  def logh(x):
    upper = jnp.where(x > -1.0, x, -1.0)
    lower = jnp.where(x <= -1.0, x, -1.0)

    return jnp.where(x > -1, log_ei_upper(upper), log_ei_lower(lower))

  u = scaled_improvement(loc, scale, best)
  return logh(u) + jnp.log(scale)


def ucb(loc: Array, scale: Array, beta: Numeric) -> Array:
  return loc + jnp.sqrt(beta) * scale
