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

from typing import Callable

from jax import jit, vmap

from boax.core.prediction.models.kernels.base import Kernel
from boax.utils.typing import Numeric


def from_kernel_function(
  kernel_fn: Callable[[Numeric, Numeric], Numeric],
) -> Kernel:
  """
  Transforms a kernel function into a _kernel_.

  Args:
    kernel_fn: The kernel function to be transformed.

  Returns:
    The corresponding `Kernel`.
  """

  return jit(vmap(vmap(kernel_fn, in_axes=(None, 0)), in_axes=(0, None)))
