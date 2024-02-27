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

"""Construct predictions."""

from typing import Any, Callable, TypeVar

from boax.prediction.models.base import Model
from boax.prediction.objectives.base import Objective
from boax.utils.functools import identity
from boax.utils.typing import Array, Numeric

T = TypeVar('T')


def construct(
  model_fn: Callable[[Any], Model[T]],
  objective_fn: Callable[[Any], Objective[T]],
  projection_fn: Callable[[Any], Any] = identity,
) -> Callable[[Any, Array, Array], Numeric]:
  """
  Constructs a loss function.

  Args:
    model_fn: The model function.
    objective_fn: The objective function.
    projection_fn: The projection function.

  Returns:
    The loss function.
  """

  def loss_fn(params, x, y):
    projected = projection_fn(params)

    return objective_fn(projected)(model_fn(projected)(x), y)

  return loss_fn
