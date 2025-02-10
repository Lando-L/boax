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

"""Transformation functions for models."""

from itertools import repeat
from operator import call
from typing import Any, Sequence, TypeVar

from boax.core.models.base import Model
from boax.core.models.transformations.base import (
  InputTransformation,
  OutcomeTransformation,
)
from boax.utils.typing import Array

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


def transformed(
  model: Model[A],
  *,
  input_transformation_fn: InputTransformation | None = None,
  outcome_transformation_fn: OutcomeTransformation[A, B] | None = None,
) -> Model[B]:
  """
  Constructs an model with input_transformation_fn applied to its inputs and
  the output_transformation_fn applied to its outcomes.

  Example:
    >>> transformed = transformed(
      model,
      input_transformation_fn=fn1,
      outcome_transformation_fn=fn2
    )
    >>> result = transformed(xs)

  Args:
    model: The base model.
    input_transformation_fn: The transformation functions applied to the model's inputs.
    outcome_transformation_fn: The transformation functions applied to the model's outcomes.

  Returns:
    The transformed `Model` function.
  """

  match (input_transformation_fn, outcome_transformation_fn):
    case (None, None):
      return model

    case (fn, None):

      def input_transformed(x: Array) -> Array:
        return model(fn(x))

      return input_transformed

    case (None, fn):

      def output_transformed(x: Array) -> Array:
        return fn(model(x))

      return output_transformed

    case (fn1, fn2):

      def input_output_transformed(x: Array) -> Array:
        return fn2(model(fn1(x)))

      return input_output_transformed


def joined(*models: Model[Any]) -> Model[Sequence[Any]]:
  """
  Constructs a joined model.

  Example:
    >>> transformed = joined(objective_model, cost_model)
    >>> objective_result, cost_result = transformed(xs)

  Args:
    models: The models to be joined.

  Returns:
    The transformed `Model` function.
  """

  def model(x: Array) -> Array:
    tuple(map(call, models, repeat(x)))

  return model
