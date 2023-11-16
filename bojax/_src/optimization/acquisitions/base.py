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

"""Base interface for acquisition functions."""

from typing import Protocol

from bojax._src.typing import Array


class Acquisition(Protocol):
  """Base interface for acquisition functions."""

  def __call__(self, candidates: Array, **kwargs) -> Array:
    """
    Evaluates the acquisition function on a set of `candidates`.

    Args:
      candidates: The candidate set.
      kwargs: Additional keyword arguments

    Returns:
      The acquisition function values of the given set of candidates.
    """
