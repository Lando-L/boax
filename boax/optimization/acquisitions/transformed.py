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

"""Transformation functions for kernels."""

from jax import lax

from boax.optimization.acquisitions.base import Acquisition
from boax.optimization.constraints.base import Constraint
from boax.utils.functools import combine


def constrained(
  acqusition: Acquisition, *constraints: Constraint
) -> Acquisition:
  """
  Constrains a given acquisition function.

  Example:
    >>> ei = expected_improvement(surrogate, best=0.2)
    >>> c1 = less_or_equal(cost, 0.8)
    >>> c2 = less_or_equal(complexity, 8.0)
    >>> c3 = greater_or_equal(complexity, 2.0)
    >>> acqf = constrained(ei, c1, c2, c3)

  Args:
    acquisition: An acquisition function to be constrained.
    *constraints: The constraints to apply.

  Returns:
    The constrained `Acquisition`.
  """

  return combine(lax.mul, 1.0, acqusition, *constraints)


def log_constrained(
  acqusition: Acquisition, *constraints: Constraint
) -> Acquisition:
  """
  Constrains a given log acquisition function.

  Example:
    >>> ei = log_expected_improvement(surrogate, best=0.2)
    >>> c1 = log_less_or_equal(cost, 0.8)
    >>> c2 = log_less_or_equal(complexity, 8.0)
    >>> c3 = log_greater_or_equal(complexity, 2.0)
    >>> acqf = log_constrained(ei, c1, c2, c3)

  Args:
    acquisition: An acquisition function to be constrained.
    *constrains: The constrains to apply.

  Returns:
    The constrained `Acquisition`.
  """

  return combine(lax.add, 0.0, acqusition, *constraints)
