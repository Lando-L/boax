# Copyright 2023 The Bojax Authors.
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

"""Alias for kernels."""

from jax import numpy as jnp

from bojax._src.prediction.kernels.base import Kernel
from bojax._src.typing import Array, Numeric


def squared_distance(x: Array, y: Array) -> Array:
  """
  Computes the squared distance between vectors `x` and `y`.

  Args:
    x: A vector.
    y: A vector.

  Returns:
    Squared distance.
  """

  return jnp.sum(x**2) + jnp.sum(y**2) - 2 * jnp.inner(x, y)


def rbf(length_scale: Numeric) -> Kernel:
  """
  Radial basis function (RBF) kernel.

  Computes `k(x, y) = exp(-||x - y||**2 / (2 * length_scale**2))`.

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A RBF `Kernel`.
  """

  def kernel(x, y):
    return jnp.exp(-0.5 * squared_distance(x / length_scale, y / length_scale))

  return kernel


def matern_one_half(length_scale: Numeric) -> Kernel:
  """
  Matern kernel with parameter 1/2.

  Computes `k(x, y) = exp(-||x - y|| / length_scale)`.

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern one half `Kernel`.
  """

  def kernel(x, y):
    return jnp.exp(
      -jnp.sqrt(squared_distance(x / length_scale, y / length_scale))
    )

  return kernel


def matern_three_halves(length_scale: Numeric) -> Kernel:
  """
  Matern kernel with parameter 3/2.

  Computes `k(x, y) = (1 + z) * exp(-z)`,

  with `z = sqrt(3) * ||x - y|| / length_scale`.

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern three halves `Kernel`.
  """

  def kernel(x, y):
    K = jnp.sqrt(3) * jnp.sqrt(
      squared_distance(x / length_scale, y / length_scale)
    )
    K = (1.0 + K) * jnp.exp(-K)
    return K

  return kernel


def matern_five_halves(length_scale: Numeric) -> Kernel:
  """
  Matern kernel with parameter 5/2.

  Computes `k(x, y) = (1 + z + z**2 / 3) * exp(-z)`,

  with `z = sqrt(5) * ||x - y|| / length_scale`.

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern five halves `Kernel`.
  """

  def kernel(x, y):
    K = jnp.sqrt(5) * jnp.sqrt(
      squared_distance(x / length_scale, y / length_scale)
    )
    K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
    return K

  return kernel


def periodic(
  length_scale: Numeric, variance: Numeric, period: Numeric
) -> Kernel:
  """
  Periodic kernel.

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.
    variance: The parameter controlling the variance of the kernel.
    period: The parameter controlling the period of the kernel.

  Returns:
    A periodic `Kernel`.
  """

  def kernel(x, y):
    sine_squared = (jnp.sin(jnp.pi * (x - y) / period) / length_scale) ** 2
    K = variance * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
    return K

  return kernel
