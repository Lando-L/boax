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

"""Transformation functions for kernels."""

from functools import partial

from jax import lax
from jax import numpy as jnp

from boax.prediction.kernels.base import Kernel
from boax.utils.functools import combine, compose
from boax.utils.typing import Array, Numeric


def scaled(kernel: Kernel, amplitude: Numeric) -> Kernel:
  """
  Scales a given kernel.

  Computes `k(x, y) = amplitude * inner(x, y)`.

  Example:
    >>> inner = rbf(jnp.array([0.2, 3.0]))
    >>> kernel = scale(inner, 3.0)
    >>> Kxx = kernel(xs, xs)

  Args:
    kernel: The kernel to be scaled.
    amplitude: The parameter controlling the maximum of the kernel.

  Returns:
    A scaled `Kernel`.
  """

  return compose(partial(lax.mul, y=amplitude), kernel)


def linear_truncated(
  x_fidelities: Array,
  y_fidelities: Array,
  unbiased: Kernel,
  biased: Kernel,
  power: Numeric,
) -> Kernel:
  """
  Constructs a linear truncated kernel for one fidelity parameter.

  Computes `k(x, y) = k_0(x, y) + c(x_fid, y_fid) k_1(x, y)`,

  where `k_i(x, y)` are Matern kernels calculated between `x` and `y`, and
  `c(x_fid, y_fid) = (1 - x_fid)(1 - y_fid))(1 + x_fid y_fid)^p`.

  Example:
    >>> unbiased = matern_five_halves(jnp.array([0.2, 3.0]))
    >>> biased = matern_five_halves(jnp.array([1.5]))
    >>> kernel = linear_truncated(x_fid, y_fid, unbiased, biased, 1.0)
    >>> Kxx = kernel(xs, xs)

  Args:
    x_fidelities: The fidelity parameters of `x`.
    y_fidelities: The fidelity parameters of `y`.
    unbiased: The unbiased kernel `k_0`.
    biased: The biased kernel `k_1`.
    power: The order of the polynomial kernel `p`.

  Returns:
    A linear truncated `Kernel`.
  """

  bias_factor = (
    (1 - x_fidelities)
    * (1 - y_fidelities.T)
    * jnp.power(1 + x_fidelities * y_fidelities.T, power)
  )

  return combine(
    lax.add, 0.0, unbiased, compose(partial(lax.mul, y=bias_factor), biased)
  )


def additive(*kernels: Kernel) -> Kernel:
  """
  Constructs an additive kernel which sums over a sequence of kernels.

  Computes `k(x, y) = k1(x, y) + k2(x, y) + ... + kn(x, y)`.

  Example:
    >>> kernel = additive(map(rbf, [0.2, 0.3, 0.4]))
    >>> Kxx = kernel(xs, xs)

  Args:
    kernels: The sequence of kernels to sum.

  Returns:
    An additive `Kernel`.
  """

  return combine(lax.add, 0.0, *kernels)


def product(*kernels: Kernel) -> Kernel:
  """
  Constructs a product kernel which multiplies over a sequence of kernels.

  Computes `k(x, y) = k1(x, y) * k2(x, y) * ... * kn(x, y)`.

  Example:
    >>> kernel = product(map(rbf, [0.2, 0.3, 0.4]))
    >>> Kxx = kernel(xs, xs)

  Args:
    kernels: The sequence of inner kernels.

  Returns:
    A product `Kernel`.
  """

  return combine(lax.mul, 1.0, *kernels)
