from operator import itemgetter

from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import optimizers


class SolversTest(parameterized.TestCase):
  def test_scipy(self):
    key = random.key(0)
    n, q, d = 10, 3, 1

    acqf = itemgetter((..., 0, 0))
    bounds = jnp.array([[-1.0, 1.0]])
    candidates = random.uniform(
      key, minval=bounds[:, 0], maxval=bounds[:, 1], shape=(n, q, d)
    )

    next_candidates, values = optimizers.solvers.scipy()(
      acqf,
      bounds,
      candidates,
    )

    self.assertEqual(next_candidates.shape, (n, q, d))
    self.assertEqual(values.shape, (n,))


if __name__ == '__main__':
  absltest.main()
