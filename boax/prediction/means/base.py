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

from boax.typing import Array


class Mean(Protocol):
  """Base interface for mean functions."""

  def __call__(self, value: Array) -> Array:
    """
    Returns the mean.

    Args:
      value: Input to the mean function.

    Returns:
      The mean.
    """
