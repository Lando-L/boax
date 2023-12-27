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

from typing import Protocol

from boax.utils.typing import Array, PRNGKey


class Sampler(Protocol):
  """
  A callable type for sampling functions.

  A sampler takes a PRNG key and a number of results as input
  and returns `num_results` samples.
  """

  def __call__(self, key: PRNGKey, num_results: int) -> Array:
    """
    Draws `num_results` of samples.

    Args:
      key: The pseudo-random number generator key.
      candidates: The number of results to return.

    Returns:
      A set of `num_results` samples.
    """
