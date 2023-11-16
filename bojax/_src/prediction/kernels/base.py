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

"""Base interfaces for kernels."""

from typing import Protocol

from jax import vmap

from bojax._src.typing import Array


class Kernel(Protocol):
  """Base interfaces for kernels."""

  def __call__(self, x: Array, y: Array) -> Array:
    """
    Calculates kernel function to pairs of inputs.

    Args:
      x: A vector.
      y: A vector.

    Returns:
      The value of the kernel function.
    """


def kernel(kernel_fn: Kernel) -> Kernel:
  """
  Transforms a single value kernel function into a batched kernel function.

  Args:
    kernel_fn: The single value kernel function.

  Returns:
    The batched kernel function.
  """

  return vmap(vmap(kernel_fn, in_axes=(None, 0)), in_axes=(0, None))
