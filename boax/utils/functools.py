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

"""The functools sub-package."""

from functools import reduce
from typing import Callable, TypeVar

T = TypeVar('T')


def identity(i: T) -> T:
  """
  Identity Function

  Args:
    i: Input value.

  Returns:
    The input value.
  """

  return i


def const(c: T) -> Callable:
  """
  Constant Function

  Args:
    c: Constant value.

  Returns:
    A function that returns the given constant.
  """

  def __fn(*args, **kwargs) -> T:
    return c

  return __fn


def call(*args, **kwargs) -> Callable:
  """ """

  def __fn(fn: Callable):
    return fn(*args, **kwargs)

  return __fn


def compose(*fns: Callable) -> Callable:
  """
  Composes a sequence of functions

  Args:
    fns: The functions to compose.

  Returns:
    A function composed of the sequential applications of the given functions.
  """

  def __reduce_fn(f: Callable, g: Callable) -> Callable:
    def __fn(*args, **kwargs):
      return f(g(*args, **kwargs))

    return __fn

  return reduce(__reduce_fn, fns)


def combine(operator: Callable, initial: T, *fns: Callable) -> Callable:
  """
  Combines a sequence of functions by an operator.

  Args:
    operator: The operator used to combine the functions.
    fns: The functions to combine.

  Returns:
    A function combining the outputs of the given functions by the operator.
  """

  def __fn(*args, **kwargs):
    def __reduce_fn(state: T, f: Callable) -> T:
      return operator(state, f(*args, **kwargs))

    return reduce(__reduce_fn, fns, initial)

  return __fn


def tupled(f: Callable) -> Callable:
  """
  Transforms function to take a tuple as its argument.

  Args:
    fns: Functions to transform.

  Returns:
    The transformed function.
  """

  def __fn(x):
    return f(*x)

  return __fn


def untupled(f: Callable) -> Callable:
  """
  Transforms function to take a regular list arguments.

  Args:
    fns: Functions to transform.

  Returns:
    The transformed function.
  """

  def __fn(*xs):
    return f(tuple(xs))

  return __fn
