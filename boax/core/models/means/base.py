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

"""Base interface for mean functions."""

from typing import Protocol

from boax.utils.typing import Array


class Mean(Protocol):
  """
  A callable type for mean functions.

  A mean functions takes a `b0 x ... x bn x n x d`-dim array as input and returns
  an `b0 x ... x bn x n`-dim array of mean function values.
  """

  def __call__(self, index_points: Array) -> Array:
    """
    Calculates the mean function to a set of inputs.

    Args:
      index_points: The inputs to the mean function.

    Returns:
      The function values at the inputs.
    """
