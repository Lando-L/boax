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

"""Transformation functions for kernels."""

from typing import Callable

from jax import numpy as jnp

from bojax._src.prediction.kernels.base import Kernel
from bojax._src.typing import Numeric


def scale(amplitude: Numeric, inner: Kernel) -> Kernel:
  """
  Scale kernel.

  Computes `k(x, y) = amplitude * inner(x, y)`.

  Args:
    amplitude: The parameter controlling the maximum of the kernel.
    inner: The inner kernel.

  Returns:
    A scaled `Kernel`.
  """

  def kernel(x, y):
    return amplitude * inner(x, y)

  return kernel


def combine(operator: Callable, *kernels: Kernel) -> Kernel:
  """
  Combine kernel.

  Combines a sequence of kernels using the an `operator`.

  Args:
    operator: The operator used for the combination.
    kernels: The sequence of inner kernels.

  Returns:
    A combined `Kernel`.
  """

  def kernel(x, y):
    return operator(jnp.stack([k(x, y) for k in kernels]))

  return kernel
