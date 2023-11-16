# Copyright 2023 The Bojax Authors.
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

"""General util functions."""

from functools import reduce
from typing import Callable, TypeVar

T = TypeVar('T')


def identity(i: T) -> T:
  return i


def const(c: T) -> Callable:
  def __fn(_) -> T:
    return c

  return __fn


def compose(*fns: Callable) -> Callable:
  def __reduce_fn(f: Callable, g: Callable) -> Callable:
    def __fn(*args, **kwargs):
      return f(g(*args, **kwargs))

    return __fn

  return reduce(__reduce_fn, fns)
