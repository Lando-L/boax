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

"""IID sampling functions."""

from typing import Sequence

from jax import random

from boax.utils.typing import PRNGKey, Array


def uniform(key: PRNGKey, sample_shape: Sequence[int], ndims: int) -> Array:
  return random.uniform(key, sample_shape + (ndims,))


def normal(key: PRNGKey, sample_shape: Sequence[int], ndims: int) -> Array:
  return random.normal(key, sample_shape + (ndims,))
