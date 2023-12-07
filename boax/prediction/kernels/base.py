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

"""Base interfaces for kernels."""

from typing import Protocol

from boax.typing import Array


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
