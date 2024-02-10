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
from typing import Any, Callable, Tuple, TypeVar

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')


def identity(i: T) -> T:
  """
  Identity Function

  Args:
    i: Input value.

  Returns:
    The input value.
  """

  return i


def const(c: T) -> Callable[[Any], T]:
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


def call(*args, **kwargs) -> Callable[[Callable[[Any], T]], T]:
  """
  Calls a callable with the given inputs.

  Args:
    *args: The arguments.
    **kwargs: The keyword arguments.

  Returns:
    A function that calls given function with the inputs.
  """

  def __fn(fn: Callable[[Any], T]) -> T:
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


def combine(
  operator: Callable[[T, T], T], initial: T, *fns: Callable[[Any], T]
) -> Callable[[Any], T]:
  """
  Combines a sequence of functions by an operator.

  Args:
    operator: The operator used to combine the functions.
    initial: The initial value.
    fns: The functions to combine.

  Returns:
    A function combining the outputs of the given functions by the operator.
  """

  def __fn(*args, **kwargs):
    def __reduce_fn(state: T, f: Callable) -> T:
      return operator(state, f(*args, **kwargs))

    return reduce(__reduce_fn, fns, initial)

  return __fn


def apply(operator: Callable, *fns: Callable) -> Callable:
  """
  Applies a sequence of functions by an operator.

  Args:
    operator: The operator used to apply the functions.
    fns: The functions to apply.

  Returns:
    A function applying the operator on the outputs of the given functions.
  """

  def __fn(*args, **kwargs):
    return operator(map(call(*args, **kwargs), fns))

  return __fn


def sequence(operator: Callable[[T, T], T], initial: T, *tuples) -> T:
  """
  Combines the output of a sequence of functions to a sequence of inputs.

  Args:
    operator: The operator used to combine the function outputs.
    initial: The initial value.
    tuples: The functions and inputs.

  Returns:
    The combined output of the given functions and inputs.
  """

  def __reduce_fn(state: T, x: Tuple[Callable[[Any], T], Any]):
    return operator(state, x[0](x[1]))

  return reduce(__reduce_fn, tuples, initial)


def wrap(fn: Callable) -> Callable:
  """
  Transforms a fuction to take a sequence of input parameters.

  Args:
    fn: The functions to be wrapped.

  Returns:
    A function taking a sequence of input paramters.
  """

  def __fn(*args):
    return fn(args)

  return __fn


def unwrap(fn: Callable) -> Callable:
  """
  Transforms a fuction to take a tuple of input parameters.

  Args:
    fn: The functions to be unwrapped.

  Returns:
    A function taking a tuple of input paramters.
  """

  def __fn(args):
    return fn(*args)

  return __fn
