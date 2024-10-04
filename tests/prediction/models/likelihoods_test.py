from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.prediction.models import likelihoods
from boax.utils.typing import PRNGKey


class LikelihoodsTest(parameterized.TestCase):
  @parameterized.parameters(
    {"key": random.key(0), "noise": 1e-4, "num_index_points": 10}
  )
  def test_gaussian(self, key: PRNGKey, noise: float, num_index_points: int):
    key1, key2 = random.split(key)

    mean = random.uniform(key1, (num_index_points,))
    cov = random.uniform(key2, (num_index_points,)) * jnp.identity(num_index_points)
    mvn = multivariate_normal.multivariate_normal(mean, cov)

    result = likelihoods.gaussian(noise)(mvn)
    expected = multivariate_normal.multivariate_normal(
      mean, cov + noise * jnp.identity(num_index_points)
    )

    assert_trees_all_close(result.mean, expected.mean, atol=1e-4)
    assert_trees_all_close(result.cov, expected.cov, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
