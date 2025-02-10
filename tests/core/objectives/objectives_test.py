from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core import objectives
from boax.core.distributions import multivariate_normal
from boax.utils.typing import PRNGKey, Shape


class ObjectivesTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'n': 3},
    {'key': random.key(1), 's': (4,), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'n': 3},
  )
  def test_negative_log_likelihood(
    self,
    key: PRNGKey,
    s: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    mean, variance = random.uniform(key1, (2, n))
    prediction = multivariate_normal.multivariate_normal(
      mean, variance * jnp.identity(n)
    )

    targets = multivariate_normal.sample(
      prediction, random.normal(key2, s + (n,))
    )

    objective = objectives.negative_log_likelihood(multivariate_normal.logpdf)
    result = objective(prediction, targets)

    expected = multivariate_normal.logpdf(prediction, targets)

    assert_trees_all_close(result, -expected, atol=1e-4)

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'n': 3},
    {'key': random.key(1), 's': (4,), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'n': 3},
  )
  def test_penalized(
    self,
    key: PRNGKey,
    s: Shape,
    n: int,
  ):
    key1, key2, key3 = random.split(key, 3)

    mean, variance = random.uniform(key1, (2, n))
    prediction = multivariate_normal.multivariate_normal(
      mean, variance * jnp.identity(n)
    )

    targets = multivariate_normal.sample(
      prediction, random.normal(key2, s + (n,))
    )

    objective = objectives.negative_log_likelihood(multivariate_normal.logpdf)
    penalization = random.uniform(key3, (n,))

    result = objectives.transformations.penalized(objective, *penalization)(
      prediction,
      targets,
    )

    expected = multivariate_normal.logpdf(
      prediction,
      targets,
    )

    assert_trees_all_close(result, -expected + jnp.sum(penalization), atol=1e-4)


if __name__ == '__main__':
  absltest.main()
