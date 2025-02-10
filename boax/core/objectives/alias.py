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

"""Alias for likelihoods."""

from typing import Callable, TypeVar

from jax import jit

from boax.core.objectives.base import Objective
from boax.utils.typing import Array

T = TypeVar('T')


def negative_log_likelihood(
  logpdf_fn: Callable[[T, Array], Array],
) -> Objective[T]:
  """
  The negative log likelihood objective function.

  Example:
    >>> objective = negative_log_likelihood(logpdf_fn)
    >>> nll = objective(prediction, targets)

  Args:
    logpdf_fn: The log probability mass function.

  Returns:
    The negative log likelihood `Objective` function.
  """

  def objective(prediction, targets):
    return -logpdf_fn(prediction, targets)

  return jit(objective)
