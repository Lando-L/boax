from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.core.models import likelihoods
from boax.utils.typing import PRNGKey, Shape


class LikelihoodsTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'noise': 1e-4, 'b': (), 'n': 3},
    {'key': random.key(1), 'noise': 1e-4, 'b': (), 'n': 10},
    {'key': random.key(2), 'noise': 1e-4, 'b': (5, 3), 'n': 3},
  )
  def test_gaussian(
    self,
    key: PRNGKey,
    noise: float,
    b: Shape,
    n: int,
  ):
    mean, variance = random.uniform(key, (2,) + b + (n,))
    cov = variance[..., jnp.newaxis] * jnp.broadcast_to(
      jnp.identity(n), b + (n, n)
    )
    mvn = multivariate_normal.multivariate_normal(mean, cov)

    result = likelihoods.gaussian(noise)(mvn)

    expected = multivariate_normal.multivariate_normal(
      mean, cov + noise * jnp.identity(n)
    )

    assert_trees_all_close(result.mean, expected.mean, atol=1e-4)
    assert_trees_all_close(result.cov, expected.cov, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
