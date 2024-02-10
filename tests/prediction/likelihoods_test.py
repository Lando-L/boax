import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.prediction import likelihoods


class LikelihoodsTest(parameterized.TestCase):
  def test_gaussian(self):
    key1, key2 = random.split(random.key(0))

    mean = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    mvn = multivariate_normal.multivariate_normal(mean, cov)

    result = likelihoods.gaussian(1e-4)(mvn)
    expected = multivariate_normal.multivariate_normal(
      mean, cov + 1e-4 * jnp.identity(10)
    )

    np.testing.assert_allclose(result.mean, expected.mean, atol=1e-4)
    np.testing.assert_allclose(result.cov, expected.cov, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
