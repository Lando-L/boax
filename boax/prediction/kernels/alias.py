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

"""Alias for kernels."""

from functools import partial

from boax.prediction.kernels import functions
from boax.prediction.kernels.base import Kernel, from_kernel_function
from boax.utils.typing import Array, Numeric


def rbf(length_scale: Array) -> Kernel:
  """
  The Radial basis function (RBF) kernel.

  Computes `k(x, y) = exp(-||x - y||**2 / (2 * length_scale**2))`.

  Example:
    >>> kernel = rbf(jnp.array([0.2, 3.0]))
    >>> Kxx = kernel(xs, xs)

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A RBF `Kernel`.
  """

  return from_kernel_function(
    partial(functions.rbf.rbf, length_scale=length_scale)
  )


def matern_one_half(length_scale: Array) -> Kernel:
  """
  The Matern kernel with parameter 1/2.

  Computes `k(x, y) = exp(-||x - y|| / length_scale)`.

  Example:
    >>> kernel = matern_one_half(jnp.array([0.2, 3.0]))
    >>> Kxx = kernel(xs, xs)

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern one half `Kernel`.
  """

  return from_kernel_function(
    partial(functions.matern.one_half, length_scale=length_scale)
  )


def matern_three_halves(length_scale: Array) -> Kernel:
  """
  The Matern kernel with parameter 3/2.

  Computes `k(x, y) = (1 + z) * exp(-z)`,

  with `z = sqrt(3) * ||x - y|| / length_scale`.

  Example:
    >>> kernel = matern_three_halves(jnp.array([0.2, 3.0]))
    >>> Kxx = kernel(xs, xs)

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern three halves `Kernel`.
  """

  return from_kernel_function(
    partial(functions.matern.three_halves, length_scale=length_scale)
  )


def matern_five_halves(length_scale: Array) -> Kernel:
  """
  The Matern kernel with parameter 5/2.

  Computes `k(x, y) = (1 + z + z**2 / 3) * exp(-z)`,

  with `z = sqrt(5) * ||x - y|| / length_scale`.

  Example:
    >>> kernel = matern_five_halves(jnp.array([0.2, 3.0]))
    >>> Kxx = kernel(xs, xs)

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.

  Returns:
    A matern five halves `Kernel`.
  """

  return from_kernel_function(
    partial(functions.matern.five_halves, length_scale=length_scale)
  )


def periodic(length_scale: Array, variance: Numeric, period: Numeric) -> Kernel:
  """
  The Periodic kernel.

  Example:
    >>> kernel = periodic(jnp.array([0.2, 3.0]), 1.0, 2.0)
    >>> Kxx = kernel(xs, xs)

  Args:
    length_scale: The parameter controlling how sharp or wide the kernel is.
    variance: The parameter controlling the variance of the kernel.
    period: The parameter controlling the period of the kernel.

  Returns:
    A periodic `Kernel`.
  """

  return from_kernel_function(
    partial(
      functions.periodic.periodic,
      length_scale=length_scale,
      variance=variance,
      period=period,
    )
  )
