from operator import itemgetter

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.optimization import optimizers


class OptimizersTest(parameterized.TestCase):
  def test_batch(self):
    key = random.key(0)

    fun = itemgetter((..., 0, 0))
    n, q, d = 10, 3, 1

    initializer = lambda k: random.uniform(k, (n, q, d))
    solver = lambda c: (c, fun(c))
    optimizer = optimizers.batch(initializer, solver)

    next_x, next_v = optimizer(key)

    assert_shape(next_x, (q, d))
    assert_shape(next_v, ())

  def test_sequential(self):
    key = random.key(0)

    fun = itemgetter((..., 0, 0))
    n, q, d = 10, 3, 1

    initializer = lambda k: random.uniform(k, (n, 1, d))
    solver = lambda c: (c, fun(c))
    optimizer = optimizers.sequential(initializer, solver, q)

    next_x, next_v = optimizer(key)

    assert_shape(next_x, (q, d))
    assert_shape(next_v, (q,))


if __name__ == '__main__':
  absltest.main()
