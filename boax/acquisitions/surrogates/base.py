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

"""Base interface for surrogate model functions."""

from typing import Generic, NamedTuple, Protocol, TypeVar

from boax.core.models import Model
from boax.utils.typing import Array, PRNGKey

T = TypeVar('T')
S = TypeVar('S')


class InitFn(Protocol, Generic[T]):
  def __call__(self) -> T:
    pass


class UpdateFn(Protocol, Generic[T]):
  def __call__(
    self, params: T, observation_index_points: Array, observations: Array
  ) -> T:
    pass


class PriorFn(Protocol, Generic[T, S]):
  def __call__(self, params: T) -> Model[S]:
    pass


class PosteriorFn(Protocol, Generic[T, S]):
  def __call__(
    self, params: T, observation_index_points: Array, observations: Array
  ) -> Model[S]:
    pass


class BestFn(Protocol, Generic[S]):
  def __call__(self, key: PRNGKey, model: Model[S]) -> tuple[Array, float]:
    pass


class Surrogate(NamedTuple, Generic[T, S]):
  init: InitFn[T]
  update: UpdateFn[T]
  prior: PriorFn[T, S]
  posterior: PosteriorFn[T, S]
  best: BestFn[S]
