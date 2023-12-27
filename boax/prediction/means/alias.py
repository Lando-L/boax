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

"""Alias for mean functions."""

from jax import jit, vmap
from jax import numpy as jnp

from boax.prediction.means.base import Mean
from boax.utils.functools import const
from boax.utils.typing import Array, Numeric


def zero() -> Mean:
  """
  The zero mean function.

  Computes `y = f(x) = 0`.

  Example:
    >>> mean = zero()
    >>> ys = mean(xs)

  Returns:
    A zero mean function.
  """

  return jit(vmap(const(jnp.zeros(()))))


def constant(x: Numeric) -> Mean:
  """
  The constant mean function.

  Computes `y = f(x) = c`.

  Example:
    >>> mean = constant(1.96)
    >>> ys = mean(xs)

  Returns:
    A constant mean function.
  """

  return jit(vmap(const(x)))


def linear(scale: Array, bias: Numeric) -> Mean:
  """
  The linear mean function.

  Computes `y = f(x) = scale * x + bias`.

  Example:
    >>> mean = linear(jnp.array([0.2, 3.0]), 5.0)
    >>> ys = mean(xs)

  Returns:
    A linear mean function.
  """

  def mean(value: Array) -> Array:
    return jnp.dot(scale, value) + bias

  return jit(mean)
