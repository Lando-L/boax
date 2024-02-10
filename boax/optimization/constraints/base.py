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

"""Base interface for constraint functions."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array

T = TypeVar('T')


class Constraint(Protocol, Generic[T]):
  """
  A callable type for constraint functions.

  An acquisition function takes a posterior prediction of type `T`
  as input and returns a numeric feasiblity score.
  """

  def __call__(self, T) -> Array:
    """
    Evaluates the constrained function on a posterior prediction.

    Args:
      posterior: The posterrior prediction.

    Returns:
      The `n`-dim feasibility scores.
    """
