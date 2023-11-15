from functools import reduce
from typing import Callable, TypeVar

T = TypeVar("T")


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
