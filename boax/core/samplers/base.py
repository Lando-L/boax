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

"""Base interface for samplers."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array

T = TypeVar('T')


class Sampler(Protocol, Generic[T]):
  """
  A callable type for sampling functions.

  A sampler takes a distribution as its input and samples from it.
  """

  def __call__(self, distribution: T) -> Array:
    """
    Draws samples from the given distribution.

    Args:
      distribution: The distribution to be sampled from.

    Returns:
      A set of samples.
    """
