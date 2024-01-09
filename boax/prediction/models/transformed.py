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

from boax.prediction.likelihoods.base import Likelihood
from boax.prediction.models.base import Model
from boax.utils.functools import compose
from boax.utils.typing import Array

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')


def sampled(
  model: Model[T], sample_fn: Callable[[T, Array], Array]
) -> Model[Callable[[Array], Array]]:
  return compose(
    partial(partial, sample_fn),
    model,
  )


def predictive(
  model: Model[A],
  likelihood_fn: Likelihood[A, B],
) -> Model[B]:
  return compose(
    likelihood_fn,
    model,
  )


def fantisized(
  model: Model[T], fantasy_fn: Callable[[T, Array], Array]
) -> Model[Callable[[Array], Array]]:
  return
