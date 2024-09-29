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

"""Base interface for policy functions."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Numeric, PRNGKey

T = TypeVar('T')


class Policy(Protocol, Generic[T]):
  """
  A callable type for policy functions.

  A policy function takes a set of parameters of type `T`,
  a timestep, and a pseudo-random key as input and returns a selected variant.
  """

  def __call__(self, params: T, timestep: Numeric, key: PRNGKey) -> Numeric:
    """
    Select the variant.

    Args:
      params: The policy's parameters.
      timestep: The current timestep.
      key: A pseudo-random key.

    Returns:
      The selected variant.
    """
