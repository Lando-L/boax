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

"""Outcome transformation functions."""

from collections.abc import Callable
from functools import partial
from typing import TypeVar

from jax import numpy as jnp

from boax.core.prediction.models.transformations.base import (
  OutcomeTransformation,
)
from boax.utils.typing import Array

T = TypeVar('T')


def scaled(
  loc: Array, variance: Array, scale_fn: Callable[[T], T]
) -> OutcomeTransformation[T, T]:
  return partial(scale_fn, loc=loc, scale=jnp.sqrt(variance))
