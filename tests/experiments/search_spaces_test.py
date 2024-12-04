import math
from itertools import product
from typing import Any

from absl.testing import absltest, parameterized
from jax import numpy as jnp

from boax.experiments import search_spaces
from boax.experiments.search_spaces import SearchSpace
from boax.experiments.search_spaces.parameters import Choice, LogRange, Range


class SearchSpacesTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'choice_parameters': [
        {'type': 'choice', 'name': 'variants', 'values': [1, 2, 3]},
      ],
      'fixed_parameters': [{'type': 'fixed', 'name': 'seed', 'value': 42}],
      'range_parameters': [
        {'type': 'log_range', 'name': 'learning_rate', 'bounds': [1e-4, 1e-3]},
        {'type': 'range', 'name': 'x', 'bounds': [-1, 1]},
      ],
    },
    {
      'choice_parameters': [
        {'type': 'choice', 'name': 'variants', 'values': [1, 2, 3]},
      ],
      'fixed_parameters': [{'type': 'fixed', 'name': 'seed', 'value': 42}],
      'range_parameters': [],
    },
  )
  def test_from_dicts_successfully_parses_valid_parameters(
    self,
    choice_parameters: list[dict],
    fixed_parameters: list[dict],
    range_parameters: list[dict],
  ):
    search_space = search_spaces.from_dicts(
      choice_parameters + fixed_parameters + range_parameters
    )

    for parameter, expected in zip(
      search_space.choice_parameters, choice_parameters
    ):
      self.assertEqual(parameter.name, expected['name'])
      self.assertEqual(parameter.values, expected['values'])

    for parameter, expected in zip(
      search_space.fixed_parameters, fixed_parameters
    ):
      self.assertEqual(parameter.name, expected['name'])
      self.assertEqual(parameter.value, expected['value'])

    for parameter, expected in zip(
      search_space.range_parameters, range_parameters
    ):
      self.assertEqual(parameter.name, expected['name'])
      self.assertEqual(parameter.bounds, expected['bounds'])

  @parameterized.parameters(
    {
      'choice_parameters': [
        {'type': 'choice', 'name': 'variants', 'values': [1, 2, 3]},
      ],
      'fixed_parameters': [{'type': 'fixed', 'name': 'seed', 'value': 42}],
      'range_parameters': [
        {'type': 'other', 'name': 'error'},
      ],
    },
  )
  def test_from_dicts_rejects_invalid_parameters(
    self,
    choice_parameters: list[dict],
    fixed_parameters: list[dict],
    range_parameters: list[dict],
  ):
    self.assertIsNone(
      search_spaces.from_dicts(
        choice_parameters + fixed_parameters + range_parameters
      )
    )

  @parameterized.parameters(
    {
      'bounds': [0, 1],
    },
    {
      'bounds': [-1, 1],
    },
  )
  def test_get_bounds_successfully_retrieves_range_bounds(
    self, bounds: list[float]
  ):
    search_space = SearchSpace([], [], [Range('x', bounds)])

    lower_bound, uppder_bound = list(search_spaces.get_bounds(search_space))[0]

    self.assertEqual(lower_bound, bounds[0])
    self.assertEqual(uppder_bound, bounds[1])

  @parameterized.parameters(
    {
      'bounds': [1e-4, 1e-3],
    },
    {
      'bounds': [1, 1_000],
    },
  )
  def test_get_bounds_successfully_retrieves_log_range_bounds(
    self, bounds: list[float]
  ):
    search_space = SearchSpace([], [], [LogRange('x', bounds)])

    lower_bound, uppder_bound = list(search_spaces.get_bounds(search_space))[0]

    self.assertEqual(lower_bound, math.log(bounds[0]))
    self.assertEqual(uppder_bound, math.log(bounds[1]))

  @parameterized.parameters(
    {
      'variants': [
        ['a', 'b', 'c'],
      ],
    },
    {
      'variants': [
        ['1', '2', '3'],
        ['a', 'b'],
      ],
    },
  )
  def test_get_variants_successfully_retrieves_all_value_combinations(
    self, variants: list[list[str]]
  ):
    search_space = SearchSpace(
      [], [Choice(str(idx), values) for idx, values in enumerate(variants)], []
    )

    results = list(search_spaces.get_variants(search_space))
    expected = list(product(*variants))

    self.assertEqual(results, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'parameterization': {
        'learning_rate': 1e-3 / 2,
      },
    },
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'parameterization': {
        'learning_rate': 1e-5,
      },
    },
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'parameterization': {
        'learning_rate': 1e-2,
      },
    },
  )
  def test_from_parameterization_successfully_retrieves_log_range_values(
    self, parameters: dict[str, LogRange], parameterization: dict[str, Any]
  ):
    result = search_spaces.from_parameterizations(
      parameters.values(), parameterization
    )
    expected = [
      math.log(
        min(parameters[name].bounds[1], max(parameters[name].bounds[0], value))
      )
      for name, value in parameterization.items()
    ]

    self.assertEqual(result, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'parameterization': {
        'x': 0,
      },
    },
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'parameterization': {
        'x': -2,
      },
    },
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'parameterization': {
        'x': 2,
      },
    },
  )
  def test_from_parameterization_successfully_retrieves_range_values(
    self, parameters: dict[str, Range], parameterization: dict[str, Any]
  ):
    result = search_spaces.from_parameterizations(
      parameters.values(), parameterization
    )
    expected = [
      min(parameters[name].bounds[1], max(parameters[name].bounds[0], value))
      for name, value in parameterization.items()
    ]

    self.assertEqual(result, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'variants': Choice('variants', [1, 2, 3]),
      },
      'parameterization': {
        'variants': 2,
      },
    },
    {
      'parameters': {
        'variants': Choice('variants', [1, 2, 3]),
      },
      'parameterization': {
        'variants': 5,
      },
    },
  )
  def test_from_parameterization_successfully_retrieves_choice_values(
    self, parameters: dict[str, Choice], parameterization: dict[str, Any]
  ):
    result = search_spaces.from_parameterizations(
      parameters.values(), parameterization
    )
    expected = [
      value
      for name, value in parameterization.items()
      if value in parameters[name].values
    ]

    self.assertEqual(result, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'raw': [jnp.log(1e-3 / 2)],
    },
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'raw': [jnp.log(1e-5)],
    },
    {
      'parameters': {
        'learning_rate': LogRange('learning_rate', [1e-4, 1e-3]),
      },
      'raw': [jnp.log(1e-2)],
    },
  )
  def test_to_parameterization_successfully_retrieves_log_range_values(
    self, parameters: dict[str, LogRange], raw: list[Any]
  ):
    result = search_spaces.to_parameterizations(parameters.values(), raw)

    expected = {
      name: float(
        jnp.clip(jnp.exp(value), parameter.bounds[0], parameter.bounds[1])
      )
      for value, (name, parameter) in zip(raw, parameters.items())
    }

    self.assertEqual(result, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'raw': [jnp.array(0)],
    },
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'raw': [jnp.array(-2)],
    },
    {
      'parameters': {
        'x': Range('x', [-1, 1]),
      },
      'raw': [jnp.array(2)],
    },
  )
  def test_to_parameterization_successfully_retrieves_log_range_values(
    self, parameters: dict[str, Range], raw: list[Any]
  ):
    result = search_spaces.to_parameterizations(parameters.values(), raw)

    expected = {
      name: float(jnp.clip(value, parameter.bounds[0], parameter.bounds[1]))
      for value, (name, parameter) in zip(raw, parameters.items())
    }

    self.assertEqual(result, expected)

  @parameterized.parameters(
    {
      'parameters': {
        'variants': Choice('variants', [1, 2, 3]),
      },
      'raw': [2],
    },
    {
      'parameters': {
        'variants': Choice('variants', [1, 2, 3]),
      },
      'raw': [5],
    },
  )
  def test_to_parameterization_successfully_retrieves_choice_values(
    self, parameters: dict[str, Choice], raw: list[Any]
  ):
    result = search_spaces.to_parameterizations(parameters.values(), raw)

    expected = {
      name: value
      for value, (name, parameter) in zip(raw, parameters.items())
      if value in parameter.values
    }

    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
