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

"""Construct acquisition functions."""

from functools import partial
from typing import Callable, TypeVar

from jax import lax, vmap

from boax.optimization.acquisitions.base import Acquisition
from boax.optimization.constraints.base import Constraint
from boax.prediction.models.base import Model
from boax.utils.functools import compose, sequence, unwrap
from boax.utils.typing import Array

T = TypeVar('T')


def construct(
  model: Model[T], acquisition: Acquisition[T]
) -> Callable[[Array], Array]:
  """
  Constructs an acquisition function.

  Args:
    model: The base model.
    acquisition: The acquisition.

  Returns:
    The constructed acquisition function.
  """

  return compose(
    acquisition,
    vmap(model),
  )


def construct_constrained(
  model: Model[T], acquisition: Acquisition[T], *constraints: Constraint[T]
) -> Callable[[Array], Array]:
  """
  Constructs a constrained acquisition function.

  Args:
    model: The base model.
    acquisition: The acquisition.
    constraints: The constraints.

  Returns:
    The constructed acquisition function.
  """

  return compose(
    unwrap(partial(partial, sequence)(lax.mul, 1.0)),
    partial(partial, zip)((acquisition, *constraints)),
    vmap(model),
  )


def construct_log_constrained(
  model: Model[T], acquisition: Acquisition[T], *constraints: Constraint[T]
) -> Callable[[Array], Array]:
  """
  Constructs a log constrained acquisition function.

  Args:
    model: The base model.
    acquisition: The log acquisition.
    constraints: The log constraints.

  Returns:
    The constructed log acquisition function.
  """

  return compose(
    unwrap(partial(partial, sequence)(lax.add, 0.0)),
    partial(partial, zip)((acquisition, *constraints)),
    vmap(model),
  )
