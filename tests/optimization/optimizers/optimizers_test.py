from operator import itemgetter

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.optimization import optimizers
from boax.utils.typing import PRNGKey


class OptimizersTest(parameterized.TestCase):
  @parameterized.parameters(
    {"key": random.key(0), "n": 10, "q": 3, "d": 1},
  )
  def test_batch(self, key: PRNGKey, n: int, q: int, d: int):
    fun = itemgetter((..., 0, 0))

    initializer = lambda k: random.uniform(k, (n, q, d))
    solver = lambda c: (c, fun(c))
    optimizer = optimizers.batch(initializer, solver)

    next_x, next_v = optimizer(key)

    assert_shape(next_x, (q, d))
    assert_shape(next_v, ())

  @parameterized.parameters(
    {"key": random.key(0), "n": 10, "q": 3, "d": 1},
  )
  def test_sequential(self, key: PRNGKey, n: int, q: int, d: int):
    fun = itemgetter((..., 0, 0))

    initializer = lambda k: random.uniform(k, (n, 1, d))
    solver = lambda c: (c, fun(c))
    optimizer = optimizers.sequential(initializer, solver, q)

    next_x, next_v = optimizer(key)

    assert_shape(next_x, (q, d))
    assert_shape(next_v, (q,))


if __name__ == '__main__':
  absltest.main()
