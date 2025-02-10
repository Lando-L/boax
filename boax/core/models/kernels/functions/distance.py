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

"""Distances for kernels."""

from jax import numpy as jnp

from boax.utils.typing import Array


def squared(x: Array, y: Array) -> Array:
  """
  Computes the squared distance between vectors `x` and `y`.

  Args:
    x: A vector.
    y: A vector.

  Returns:
    Squared distance.
  """

  adjustment = jnp.mean(x, axis=-2, keepdims=True)

  x_adj = x - adjustment
  x_norm = jnp.sum(jnp.power(x_adj, 2), axis=-1, keepdims=True)
  x_pad = jnp.ones_like(x_norm)

  y_adj = y - adjustment
  y_norm = jnp.sum(jnp.power(y_adj, 2), axis=-1, keepdims=True)
  y_pad = jnp.ones_like(y_norm)

  x_ = jnp.concatenate([-2.0 * x_adj, x_norm, x_pad], axis=-1)
  y_ = jnp.concatenate([y_adj, y_pad, y_norm], axis=-1)

  distance = jnp.matmul(x_, jnp.matrix_transpose(y_))

  return jnp.maximum(distance, 0.0)


def euclidean(x: Array, y: Array) -> Array:
  """
  Computes the euclidean distance between vectors `x` and `y`.

  Args:
    x: A vector.
    y: A vector.

  Returns:
    Euclidean distance.
  """

  distance = jnp.sqrt(squared(x, y))

  return jnp.maximum(distance, 1e-30)
