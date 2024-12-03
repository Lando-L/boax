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

"""Transformation functions for acquisitions."""


from functools import partial
from typing import Sequence, TypeVar

from jax import lax

from boax.core.optimization.acquisitions.base import Acquisition
from boax.core.optimization.acquisitions.constraints.base import Constraint
from boax.utils.functools import compose, sequence, unwrap

T = TypeVar('T')


def constrained(
  acquisition: Acquisition[T], *constraints: Constraint[T]
) -> Acquisition[Sequence[T]]:
  """
  Constructs a constrained acquisition function.

  Args:
    acquisition: The acquisition.
    constraints: The constraints.

  Returns:
    The constructed acquisition function.
  """

  return compose(
    unwrap(partial(partial, sequence)(lax.mul, 1.0)),
    partial(partial, zip)((acquisition, *constraints)),
  )


def log_constrained(
  acquisition: Acquisition[T], *log_constraints: Constraint[T]
) -> Acquisition[Sequence[T]]:
  """
  Constructs a log constrained acquisition function.

  Args:
    acquisition: The log acquisition.
    constraints: The log constraints.

  Returns:
    The constructed log acquisition function.
  """

  return compose(
    unwrap(partial(partial, sequence)(lax.add, 0.0)),
    partial(partial, zip)((acquisition, *log_constraints)),
  )
