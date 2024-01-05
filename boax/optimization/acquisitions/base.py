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

"""Base interface for acquisition functions."""

from typing import Protocol

from boax.utils.typing import Array


class Acquisition(Protocol):
  """
  A callable type for acquisition functions.

  An acquisition function takes a `n x q x d`-dim candidate set as input
  and returns a numeric acquisition value.
  """

  def __call__(self, candidates: Array) -> Array:
    """
    Evaluates the acquisition function on a set of `candidates`.

    Args:
      candidates: The `n x q x d`-dim candidate set.

    Returns:
      The `n`-dim acquisition values.
    """
