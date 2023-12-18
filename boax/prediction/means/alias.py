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

from jax import jit
from jax import numpy as jnp
from jax import vmap

from boax.prediction.means.base import Mean
from boax.typing import Array, Numeric
from boax.util import const


def zero() -> Mean:
  """
  Zero function.

  Computes `y = f(x) = 0`.

  Returns:
    A zero mean function.
  """

  return jit(vmap(const(jnp.zeros(()))))


def constant(x: Numeric) -> Mean:
  """
  Constant function.

  Computes `y = f(x) = c`.

  Returns:
    A constant mean function.
  """

  return jit(vmap(const(x)))


def linear(scale: Array, bias: Numeric) -> Mean:
  """
  Linear mean function.

  Computes `y = f(x) = scale * x + bias`.

  Returns:
    A linear mean function.
  """

  def mean(value: Array) -> Array:
    return jnp.dot(scale, value) + bias

  return jit(mean)
