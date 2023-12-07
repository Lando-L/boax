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

"""Base interface for search spaces."""

from typing import NamedTuple, Protocol

from boax.typing import Array


class SearchSpaceSampleFn(Protocol):
  """A callable type for the `sample` attribute of a `SearchSpaces`."""

  def __call__(self, num_samples: int) -> Array:
    """
    Samples `num_samples` points from the search space.

    Args:
      num_samples: The number of samples.

    Returns:
      Samples from the search space.
    """


class SearchSpace(NamedTuple):
  """
  Base interface for search spaces.
  
  Attributes:
    ndims: Number of dimensions.
    bounds: Bounds of the search space.
    sample: A pure function which samples a number of points from the search space.
  """

  ndims: Array
  bounds: Array
  sample: SearchSpaceSampleFn
