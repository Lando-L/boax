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

"""Transformation functions for objective functions."""

from functools import partial
from typing import TypeVar

from jax import lax

from boax.core.prediction.objectives.base import Objective
from boax.utils.functools import compose
from boax.utils.typing import Array

T = TypeVar('T')


def penalized(
  objective: Objective[T],
  *terms: Array,
) -> Objective[T]:
  """
  Adds penalty terms to an objective function.

  Args:
    objective: The objective to be penalized.
    terms: The penlization terms added to the objective value.

  Returns:
    A penalized `Objective`.
  """

  return compose(
    partial(partial, lax.add)(sum(terms)),
    objective,
  )
