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

"""Alias for benchmark functions."""

from jax import jit
from jax import numpy as jnp

from boax.experiments.benchmarks import functions
from boax.experiments.benchmarks.base import Benchmark

beale = Benchmark(
  name='Beale',
  bounds=jnp.array([[-4.5, 4.5]] * 2),
  optima=jnp.array([3.0, 0.5]),
  objective=jit(functions.beale),
)

bohachevsky = Benchmark(
  name='Bohachevsky',
  bounds=jnp.array([[-100.0, 100.0]] * 2),
  optima=jnp.array([0, 0]),
  objective=jit(functions.bohachevsky),
)

branin = Benchmark(
  name='Branin',
  bounds=jnp.array([[-5.0, 10.0], [0.0, 15.0]]),
  optima=list[
    jnp.array([-jnp.pi, 12.275]),
    jnp.array([jnp.pi, 2.275]),
    jnp.array([9.42478, 2.475]),
  ],
  objective=jit(functions.branin),
)

forrester_1d = Benchmark(
  name='Forrester',
  bounds=jnp.array([[0.0, 1.0]]),
  optima=jnp.array([0.757249]),
  objective=jit(functions.forrester_1d),
)

grammacy_lee = Benchmark(
  name='Grammacy & Lee',
  bounds=jnp.array([[0.5, 2.5]]),
  optima=jnp.array([-0.86901113]),
  objective=jit(functions.grammacy_lee),
)

hartmann_6d = Benchmark(
  name='Hartmann 6D',
  bounds=jnp.array([[0.0, 1.0]] * 6),
  optima=jnp.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
  objective=jit(functions.hartmann_6d),
)
