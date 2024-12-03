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

from typing import Any

from jax import numpy as jnp

from boax.experiments.search_spaces.parameters.alias import (
  Choice,
  Fixed,
  LogRange,
  Parameter,
  Range,
)


def from_dict(parameter: dict) -> Parameter | None:
  """
  Parses a parameter from a dictionary.

  Example:
    >>> from_dict({'name': 'colours', 'type': 'choice', 'values': ['red', 'green', 'blue']})
    >>> from_dict({'name': 'kernel', 'type': 'fixed', 'value': 'rbf'})
    >>> from_dict({'name': 'learning_rate', 'type': 'log_range', 'bounds': [1e-4, 1e-3]})
    >>> from_dict({'name': 'click_rate', 'type': 'range', 'bounds': [0, 1]})

  Args:
    parameter: A dictionary describing the parameter.

  Returns:
    The parsed parameter or None if parameter could not be parsed.
  """

  match parameter.get('type'):
    case None:
      return None

    case 'choice' if 'name' in parameter and 'values' in parameter:
      return Choice(parameter['name'], parameter['values'])

    case 'fixed' if 'name' in parameter and 'value' in parameter:
      return Fixed(parameter['name'], parameter['value'])

    case 'log_range' if 'name' in parameter and 'bounds' in parameter:
      return LogRange(parameter['name'], parameter['bounds'])

    case 'range' if 'name' in parameter and 'bounds' in parameter:
      return Range(parameter['name'], parameter['bounds'])


def from_parameterization(
  parameter: Parameter, parameterization: dict[str, Any]
) -> Any | None:
  """
  Parses raw values from a parameterization dictionary.

  Args:
    parameter: The parameter of which the raw value should be extracted from.
    parameterization: The dictionary containing paramterization values.

  Returns:
    Parsed raw value or None if the parameter could not be found in the parameterization.
  """

  match parameter:
    case Choice(name, values) if name in parameterization and parameterization[
      name
    ] in values:
      return parameterization[name]

    case Fixed(name, value) if name in parameterization and parameterization[
      name
    ] == value:
      return value

    case LogRange(name, bounds) if name in parameterization:
      return jnp.log(jnp.clip(parameterization[name], bounds[0], bounds[1]))

    case Range(name, bounds) if name in parameterization:
      return jnp.clip(parameterization[name], bounds[0], bounds[1])

    case _:
      return None


def to_parameterization(
  parameter: Parameter, raw: Any
) -> tuple[str, Any] | None:
  """
  Parses parameterization dictionaries from raw values.

  Args:
    parameter: The parameter to use for parameterizing the raw value.
    raw: The raw value to be turned into a paramterization.

  Returns:
    Parsed parameterization dictionary given a parameter and a raw value.
  """

  match parameter:
    case Choice(name, values) if raw in values:
      return name, raw

    case Fixed(name, value) if raw == value:
      return name, raw

    case LogRange(name, bounds):
      return name, float(jnp.clip(jnp.exp(raw), bounds[0], bounds[1]))

    case Range(name, bounds):
      return name, float(jnp.clip(raw, bounds[0], bounds[1]))

    case _:
      return None
