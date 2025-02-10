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

"""Interface for parameters."""

from typing import Generic, NamedTuple, TypeVar

T = TypeVar('T')


class Choice(NamedTuple, Generic[T]):
  """
  A tuple describing a choice parameter.

  Attributes:
    name: A name describing the parameter.
    values: A list of values this parameter may take.
  """

  name: str
  values: list[T]


class LogRange(NamedTuple):
  """
  A tuple describing a logaritmhic range parameter.

  Attributes:
    name: A name describing the parameter.
    bounds: The bounds within the parameter lives.
  """

  name: str
  bounds: tuple[float, float]


class Range(NamedTuple):
  """
  A tuple describing a range parameter.

  Attributes:
    name: A name describing the parameter.
    bounds: The bounds within the parameter lives.
  """

  name: str
  bounds: tuple[float, float]


Parameter = Choice[T] | LogRange | Range
