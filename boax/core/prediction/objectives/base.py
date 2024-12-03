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

"""Base interface for objectives."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array

T = TypeVar('T')


class Objective(Protocol, Generic[T]):
  """
  A callable type for objectives.

  An objective function takes a posterior prediction of type `T`
  and an array of targets as input and returns an objective value.
  """

  def __call__(self, prediction: T, targets: Array) -> Array:
    """
    Computes the objective value at the given prediction of type `T`.

    Args:
      prediction: The prediction.
      targets: The targets.

    Returns:
      The objective values.
    """
