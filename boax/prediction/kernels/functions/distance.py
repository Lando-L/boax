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

"""Utils for kernels."""

from jax import lax
from jax import numpy as jnp

from boax.utils.typing import Array


def squared_distance(x: Array, y: Array) -> Array:
  """
  Computes the squared distance between vectors `x` and `y`.

  Args:
    x: A vector.
    y: A vector.

  Returns:
    Squared distance.
  """

  x_norm = jnp.sum(jnp.power(x, 2))
  y_norm = jnp.sum(jnp.power(y, 2))
  inner = jnp.inner(x, y)

  return lax.max(x_norm + y_norm - 2 * inner, 0.0)


def euclidean_distance(x: Array, y: Array) -> Array:
  """
  Computes the euclidean distance between vectors `x` and `y`.

  Args:
    x: A vector.
    y: A vector.

  Returns:
    Euclidean distance.
  """

  pdist = x - y
  is_zero = jnp.allclose(pdist, 0.0)
  masked_pdist = jnp.where(is_zero, jnp.ones_like(pdist), pdist)
  euclidean = jnp.linalg.norm(masked_pdist, ord=2)

  return jnp.where(is_zero, 0, euclidean)
