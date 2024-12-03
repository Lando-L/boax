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

"""Utils for parameters."""

import math
from collections.abc import Iterator
from functools import partial
from itertools import product, starmap
from operator import is_not
from typing import Any

from boax.experiments.search_spaces.base import SearchSpace
from boax.experiments.search_spaces.parameters import (
  Choice,
  Fixed,
  LogRange,
  Parameter,
  Range,
  from_dict,
  from_parameterization,
  to_parameterization,
)


def from_dicts(search_space: list[dict]) -> SearchSpace | None:
  """
  Parses a search space from a list of dictionaries each describing a single parameters.

  Args:
    search_space: A list of dictionaries each describing a single parameter.

  Returns:
    The parsed search space with its paramters or None
    if one of the dictionaries could not be parsed.
  """

  choice_parameters = []
  fixed_parameters = []
  range_parameters = []

  for parameter in search_space:
    match from_dict(parameter):
      case None:
        return None

      case Choice(name, values):
        choice_parameters.append(Choice(name, values))

      case Fixed(name, value):
        fixed_parameters.append(Fixed(name, value))

      case LogRange(name, bounds):
        range_parameters.append(LogRange(name, bounds))

      case Range(name, bounds):
        range_parameters.append(Range(name, bounds))

  return SearchSpace(fixed_parameters, choice_parameters, range_parameters)


def get_bounds(search_space: SearchSpace) -> Iterator[tuple[float, float]]:
  """
  Extracts the bounds from the search space's range parameters.

  Args:
    search_space: The search space to extract the bounds from.

  Returns:
    An iterator containing the bounds of all range parameters within the search space.
  """

  for parameter in search_space.range_parameters:
    match parameter:
      case LogRange(_, bounds):
        yield math.log(bounds[0]), math.log(bounds[1])

      case Range(_, bounds):
        yield bounds


def get_variants(search_space: SearchSpace) -> Iterator[Any]:
  """
  Extracts all different variants from the search space's choice parameters.

  Args:
    search_space: The search space to extract the bounds from.

  Returns:
    An iterator containing all combinations of the choice parameter values within the search space.
  """

  return product(*(choice.values for choice in search_space.choice_parameters))


def from_parameterizations(
  parameters: list[Parameter], parameterization: dict[str, Any]
) -> list[Any]:
  """
  Parses raw values from a parameterization dictionary.

  Args:
    parameters: A list of parameters of which the raw values should be extracted from.
    parameterization: The dictionary containing paramterization values.

  Returns:
    Parsed raw values.
  """

  return list(
    filter(
      partial(is_not, None),
      map(
        partial(
          from_parameterization,
          parameterization=parameterization,
        ),
        parameters,
      ),
    )
  )


def to_parameterizations(
  parameters: list[Parameter], raw: list[Any]
) -> dict[str, Any]:
  """
  Parses parameterization dictionaries from raw values.

  Args:
    parameter: The parameters to use for parameterizing the raw values.
    raw: The list of raw value to be turned into a paramterization.

  Returns:
    Parsed parameterization dictionary given the parameters and a raw values.
  """

  return dict(
    filter(
      partial(is_not, None),
      starmap(
        to_parameterization,
        zip(parameters, raw),
      ),
    )
  )
