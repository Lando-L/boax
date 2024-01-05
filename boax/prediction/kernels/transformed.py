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

from boax.prediction.kernels.base import Kernel
from boax.utils.functools import combine, compose
from boax.utils.typing import Numeric


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
