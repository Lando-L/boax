from operator import itemgetter

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core.optimization import optimizers
from boax.utils.typing import Array, PRNGKey


class SolversTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'n': 10,
      'q': 3,
      'd': 1,
      'bounds': jnp.array([[-1.0, 1.0]]),
    },
  )
  def test_scipy(self, key: PRNGKey, n: int, q: int, d: int, bounds: Array):
    fun = itemgetter((..., 0, 0))

    candidates = random.uniform(
      key, minval=bounds[:, 0], maxval=bounds[:, 1], shape=(n, q, d)
    )

    solver = optimizers.solvers.scipy(fun, bounds)

    next_candidates, values = solver(candidates)

    assert_shape(next_candidates, (n, q, d))
    assert_shape(values, (n,))


if __name__ == '__main__':
  absltest.main()
