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
from typing import Callable, TypeVar

from jax import vmap

from boax.prediction.likelihoods.base import Likelihood
from boax.prediction.models.base import Model
from boax.utils.functools import apply, call, compose
from boax.utils.typing import Array

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')


def predictive(
  model: Model[A], likelihood_fn: Likelihood[A, B],
) -> Model[B]:
  return compose(
    likelihood_fn,
    model,
  )


def sampled(
  model: Model[T], sample_fn: Callable[[T, Array], Array], base_samples: Array,
) -> Model[Array]:
  return compose(
    call(base_samples),
    vmap,
    partial(partial, sample_fn),
    model,
  )


def fantasized(
  model: Model[Array], fantasy_fn: Callable[[Array, Array], Model[T]], fantasy_index_points: Array,
) -> Model[T]:
  return fantasy_fn(
    fantasy_index_points,
    model(fantasy_index_points)
  )


def joined(
  *models: Model[T]
) -> Model[T]:
  return apply(
    tuple,
    *models
  )


def input_transformed(
  model: Model[T],
  transform_fn: Callable[[Array], Array],
) -> Model[T]:
  return compose(
    model,
    transform_fn,
  )


def outcome_transformed(
  model: Model[A],
  transform_fn: Callable[[A], B]
) -> Model[B]:
  return compose(
    transform_fn,
    model,
  )
