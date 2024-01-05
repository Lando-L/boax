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

"""The maximizer initialization functions."""

from jax import nn, random
from jax import numpy as jnp

from boax.core import distributions
from boax.core.samplers.alias import halton_uniform
from boax.optimization.acquisitions.base import Acquisition
from boax.utils.typing import Array, Numeric, PRNGKey


def q_batch_initialization(
  key: PRNGKey,
  acquisition: Acquisition,
  bounds: Array,
  q: int,
  num_restarts: int,
  num_raw_samples: int,
  eta: Numeric,
) -> Array:
  raw_key, sample_key = random.split(key)

  uniform = distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
  candidates = jnp.reshape(
    halton_uniform(uniform)(raw_key, num_raw_samples * q),
    (num_raw_samples, q, -1),
  )

  values = acquisition(candidates)
  weights = jnp.exp(eta * nn.standardize(values, axis=0))

  return random.choice(sample_key, candidates, (num_restarts,), p=weights)
