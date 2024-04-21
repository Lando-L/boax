from operator import itemgetter

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.optimization import optimizers


class SolversTest(parameterized.TestCase):
  def test_scipy(self):
    key = random.key(0)

    fun = itemgetter((..., 0, 0))
    bounds = jnp.array([[-1.0, 1.0]])
    n, q, d = 5, 3, 1

    candidates = random.uniform(
      key, minval=bounds[:, 0], maxval=bounds[:, 1], shape=(n, q, d)
    )

    solver = optimizers.solvers.scipy(fun, bounds)

    next_candidates, values = solver(candidates)

    assert_shape(next_candidates, (n, q, d))
    assert_shape(values, (n,))


if __name__ == '__main__':
  absltest.main()
