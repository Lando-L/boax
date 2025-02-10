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

from boax.core.models.kernels import functions
from boax.core.models.kernels.base import Kernel
from boax.utils.typing import Array


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

  return partial(
    functions.rbf.rbf,
    length_scale=length_scale,
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

  return partial(
    functions.matern.one_half,
    length_scale=length_scale,
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

  return partial(
    functions.matern.three_halves,
    length_scale=length_scale,
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

  return partial(
    functions.matern.five_halves,
    length_scale=length_scale,
  )
