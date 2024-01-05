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

"""Utils for sampling functions."""

import numpy as np


def primes_less_than(n) -> np.ndarray:
  """
  Sorted array of primes such that `2 <= prime < n`.

  Args:
    n: The upper bound for primes.

  Returns:
    The sorted array of primes.
  """

  j = 3
  primes = np.ones((n + 1) // 2, dtype=bool)

  while j * j <= n:
    if primes[j // 2]:
      primes[j * j // 2 :: j] = False
    j += 2

  ret = 2 * np.where(primes)[0] + 1
  ret[0] = 2  # :(

  return ret
