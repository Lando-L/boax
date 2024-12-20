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

"""Base interface for benchmarks."""

from typing import Callable, NamedTuple

from boax.utils.typing import Array, Numeric


class Benchmark(NamedTuple):
  """
  A tuple describing a benchmark function for bayesian optimization.

  Attributes:
    name: The name of the benchmark function.
    bounds: The bounds within the function is optimized.
    optima: The parameterizations that achive the optimum.
    objective: The benchmark function.
  """

  name: str
  bounds: Array
  optima: Array | list[Array]
  objective: Callable[[Array], Numeric]
