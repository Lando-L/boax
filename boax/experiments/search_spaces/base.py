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

from typing import NamedTuple

from boax.experiments.search_spaces.parameters.alias import (
  Choice,
  Fixed,
  LogRange,
  Range,
)


class SearchSpace(NamedTuple):
  """
  A tuple describing an experiment search space.

  Attributes:
    fixed_parameters: A list of all fixed parameters in the search space.
    choice_parameters: A list of all choice parameters in the search space.
    range_parameters: A list of all range and log_range parameters in the search space.
  """

  fixed_parameters: list[Fixed]
  choice_parameters: list[Choice]
  range_parameters: list[LogRange | Range]
