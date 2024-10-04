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

from jax import jit
from jax import numpy as jnp

from boax.core import distributions
from boax.core.distributions.normal import Normal
from boax.optimization.acquisitions.constraints.base import Constraint
from boax.utils.functools import compose
from boax.utils.typing import Numeric


def less_or_equal(
  bound: Numeric,
) -> Constraint[Normal]:
  """
  The Less or Equal unequality constraint.

  Example:
    >>> constraint = less_or_equal(0.8)
    >>> le = constraint(model(xs))

  Args:
    bound: The lower bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(distributions.normal.cdf, x=bound),
    )
  )


def log_less_or_equal(
  bound: Numeric,
) -> Constraint[Normal]:
  """
  The Log Less or Equal unequality constraint.

  Example:
    >>> constraint = log_less_or_equal(0.8)
    >>> lle = constraint(model(xs))

  Args:
    bound: The lower bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(distributions.normal.logcdf, x=bound),
    )
  )


def greater_or_equal(
  bound: Numeric,
) -> Constraint[Normal]:
  """
  The Greater or Equal unequality constraint.

  Example:
    >>> constraint = greater_or_equal(0.7)
    >>> ge = constraint(model(xs))

  Args:
    bound: The upper bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(distributions.normal.sf, x=bound),
    )
  )


def log_greater_or_equal(
  bound: Numeric,
) -> Constraint[Normal]:
  """
  The Log Greater or Equal unequality constraint.

  Example:
    >>> constraint = log_greater_or_equal(0.7)
    >>> ge = constraint(model(xs))

  Args:
    bound: The upper bound value.

  Returns:
    The corresponding `Constraint`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(distributions.normal.logsf, x=bound),
    )
  )
