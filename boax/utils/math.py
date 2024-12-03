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

"""The math sub-package."""

import math
from collections.abc import Sequence
from typing import Callable

from jax import lax, nn, scipy
from jax import numpy as jnp

from boax.utils.typing import Array, Numeric

log2 = math.log(2)

TAU = 1.0
ALPHA = 2.0

Axis = int | Sequence[int] | None


def log1mexp(x: Array) -> Array:
  upper = jnp.where(x > -log2, x, -log2)
  lower = jnp.where(x <= -log2, x, -log2)

  return jnp.where(
    x > -log2, jnp.log(-jnp.expm1(upper)), jnp.log1p(-jnp.exp(lower))
  )


def logerfcx(x: Array) -> Array:
  upper = jnp.where(x > 0.0, x, 0.0)
  lower = jnp.where(x <= 0.0, x, 0.0)

  return jnp.where(
    x > 0,
    jnp.log(jnp.exp(upper**2) * scipy.special.erfc(upper)),
    jnp.log(scipy.special.erfc(lower)) + lower**2,
  )


def logmeanexp(x: Array, *, axis: Axis, keepdims: bool = False) -> Array:
  match axis:
    case None:
      return logsumexp(x, axis=axis, keepdims=keepdims)

    case n if isinstance(n, int):
      return logsumexp(x, axis=axis, keepdims=keepdims) - math.log(x.shape[n])

    case ns:
      return logsumexp(x, axis=axis, keepdims=keepdims) - math.log(
        math.prod(x.shape[n] for n in ns)
      )


def logsumexp(x: Array, *, axis: Axis, keepdims: bool = False) -> Array:
  return _inf_max_helper(
    scipy.special.logsumexp, x=x, axis=axis, keepdims=keepdims
  )


def fatplus(x: Array, alpha: Numeric = 1e-1, tau: Numeric = TAU) -> Array:
  return tau * (nn.softplus(x / tau) + alpha * _cauchy(x / tau))


def fatmax(
  x: Array,
  *,
  axis: Axis,
  keepdims: bool = False,
  alpha: Numeric = ALPHA,
  tau: Numeric = TAU,
) -> Array:
  def max_fun(x: Array, *, axis, keepdims):
    return tau * jnp.log(
      jnp.sum(_pareto(-x, tau, alpha=alpha), axis=axis, keepdims=keepdims)
    )

  return _inf_max_helper(max_fun, x, axis=axis, keepdims=keepdims)


def _inf_max_helper(
  max_fun: Callable[[Array], Array], x: Array, *, axis: Axis, keepdims: bool
) -> Array:
  M = jnp.amax(x, axis=axis, keepdims=True)
  is_inf_max = lax.bitwise_and(M == jnp.inf, M == x)
  has_inf_max = jnp.any(is_inf_max, axis=axis, keepdims=True)

  y_inf = jnp.where(is_inf_max, x, 0.0)
  M_no_inf = jnp.where(M == jnp.inf, 0.0, M)
  y_no_inf = jnp.where(has_inf_max, 0.0, x) - M_no_inf

  result = jnp.where(
    has_inf_max,
    jnp.sum(y_inf, axis=axis, keepdims=True),
    M_no_inf + max_fun(y_no_inf, axis=axis, keepdims=True),
  )

  return result if keepdims else jnp.squeeze(result, axis=axis)


def _pareto(x: Array, *, alpha: Numeric) -> Array:
  beta_1 = alpha
  beta_0 = alpha / 2 * beta_1

  return jnp.power(
    (beta_0 / (beta_0 + beta_1 * x + jnp.square(x))),
    alpha / 2,
  )


def _cauchy(x: Array) -> Array:
  return 1 / (1 + jnp.square(x))
