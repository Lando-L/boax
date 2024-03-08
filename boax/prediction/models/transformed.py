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

from functools import partial
from typing import Callable, Sequence, TypeVar

from jax import vmap

from boax.prediction.models.base import Model
from boax.utils.functools import apply, call, compose
from boax.utils.typing import Array

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


def joined(*models: Model[T]) -> Model[Sequence[T]]:
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

  return apply(tuple, *models)


def outcome_transformed(
  model: Model[A],
  *transformation_fns: Callable[[A], B],
) -> Model[B]:
  """
  Constructs an outcome transformed model.

  Example:
    >>> transformed = outcome_transformed(model, fn1, fn2, fn3)
    >>> result = transformed(xs)

  Args:
    model: The base model.
    transformation_fns: The transformation functions.

  Returns:
    The transformed `Model` function.
  """

  return compose(
    *reversed(transformation_fns),
    model,
  )


def input_transformed(
  model: Model[A],
  *transformation_fns: Callable[[A], B],
) -> Model[B]:
  """
  Constructs an input transformed model.

  Example:
    >>> transformed = input_transformed(model, fn1, fn2, fn3)
    >>> result = transformed(xs)

  Args:
    model: The base model.
    transformation_fns: The transformation functions.

  Returns:
    The transformed `Model` function.
  """

  return compose(
    model,
    *reversed(transformation_fns),
  )


def scaled(
  model: Model[T],
  scale_fn: Callable[[T, Array, Array], T],
  loc: Array,
  scale: Array,
) -> Model[T]:
  """
  Constructs a scaled model.

  Example:
    >>> transformed = scaled(model, scale_fn, loc, scale)
    >>> result = transformed(xs)

  Args:
    model: The base model.
    scale_fn: The sampling function.
    loc: The location parameter.
    scale: The scale parameter.

  Returns:
    The transformed `Model` function.
  """

  return compose(
    partial(scale_fn, loc=loc, scale=scale),
    model,
  )


def sampled(
  model: Model[T],
  sample_fn: Callable[[T, Array], Array],
  base_samples: Array,
) -> Model[Array]:
  """
  Constructs a MC-based model.

  Example:
    >>> transformed = sampled(model, sample_fn, base_samples)
    >>> result = transformed(xs)

  Args:
    model: The base model.
    sample_fn: The sampling function.
    base_samples: The base samples of the sampling process.

  Returns:
    The transformed `Model` function.
  """

  return compose(
    call(base_samples),
    vmap,
    partial(partial, sample_fn),
    model,
  )
