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

"""Base interface for experiments."""

from typing import Any, Generic, NamedTuple, Protocol, TypeVar

T = TypeVar('T')


class Step(Generic[T], NamedTuple):
  """
  A tuple describing a single step of an experiment trial.

  Attributes:
    timestep: The step's timestep.
    state: The step's current state.
  """

  timestep: int
  state: T


class NextFn(Generic[T], Protocol):
  """
  A callable for running a single optimization step.
  """
  
  def __call__(
    self,
    step: Step[T] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[T], list[dict[str, Any]]]:
    """
    Returns the next step and a list of paramterizations to explore.

    Args:
      step: The current step object.
      results: Results from evaluating the current step parameterizations.

    Returns:
      A tuple of the next step and the paramterizations to explore at this step.
    """


class BestFn(Generic[T], Protocol):
  """
  A callable predicting the currenlty best parameterization.
  """
  def __call__(self, step: Step[T]) -> tuple[dict[str, Any], float]:
    """
    Returns the best predicted parameterization and its predicted value.

    Args:
      step: The current step object.

    Returns:
      A tuple of the best predicted parameterization and its predicted value.
    """


class Trial(Generic[T], NamedTuple):
  """
  A tuple describing an experiment trial.

  Attributes:
    next: A callable running a single optimization step.
    best: A callable predicting the currenlty best parameterization.
  """

  next: NextFn[T]
  best: BestFn[T]
