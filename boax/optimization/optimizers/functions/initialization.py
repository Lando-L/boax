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

"""Initializes candidates."""

from typing import Callable

from jax import nn, random
from jax import numpy as jnp

from boax.utils.typing import Array, Numeric, PRNGKey


def q_batch(
  key: PRNGKey,
  acquisition_fn: Callable[[Array], Array],
  x0: Array,
  num_samples: int,
  eta: Numeric = 1.0,
) -> Array:
  return random.choice(
    key,
    x0,
    (num_samples,),
    p=jnp.exp(eta * nn.standardize(acquisition_fn(x0), axis=0)),
  )
