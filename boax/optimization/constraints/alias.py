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

"""Alias for constrain functions."""

from functools import partial
from typing import Tuple

from jax import jit

from boax.optimization.constraints import functions
from boax.optimization.constraints.base import Constraint
from boax.prediction.models.base import Model
from boax.utils.functools import compose, tupled
from boax.utils.stats import mvn_to_norm
from boax.utils.typing import Array, Numeric


def less_or_equal(
  model: Model[Tuple[Array, Array]], bound: Numeric
) -> Constraint:
  """
  The Less or Equal unequality constraint.

  Example:
    >>> constraint = less_or_equal(costs, 0.8)
    >>> le = constraint(xs)

  Args:
    model: A gaussian process regression feasibility model.
    best: The lower bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      tupled(partial(functions.unequality.le, x=bound)),
      tupled(mvn_to_norm),
      model,
    )
  )


def log_less_or_equal(
  model: Model[Tuple[Array, Array]], bound: Numeric
) -> Constraint:
  """
  The Log Less or Equal unequality constraint.

  Example:
    >>> constraint = log_less_or_equal(costs, 0.8)
    >>> lle = constraint(xs)

  Args:
    model: A gaussian process regression feasibility model.
    best: The lower bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      tupled(partial(functions.unequality.lle, x=bound)),
      tupled(mvn_to_norm),
      model,
    )
  )


def greater_or_equal(
  model: Model[Tuple[Array, Array]], bound: Numeric
) -> Constraint:
  """
  The Greater or Equal unequality constraint.

  Example:
    >>> constraint = greater_or_equal(accuracy, 0.7)
    >>> ge = constraint(xs)

  Args:
    model: A gaussian process regression feasibility model.
    best: The upper bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      tupled(partial(functions.unequality.ge, x=bound)),
      tupled(mvn_to_norm),
      model,
    )
  )


def log_greater_or_equal(
  model: Model[Tuple[Array, Array]], bound: Numeric
) -> Constraint:
  """
  The Log Greater or Equal unequality constraint.

  Example:
    >>> constraint = log_greater_or_equal(accuracy, 0.7)
    >>> ge = constraint(xs)

  Args:
    model: A gaussian process regression feasibility model.
    best: The upper bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      tupled(partial(functions.unequality.lge, x=bound)),
      tupled(mvn_to_norm),
      model,
    )
  )
