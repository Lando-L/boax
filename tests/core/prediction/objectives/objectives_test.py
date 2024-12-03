from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.core.prediction import objectives
from boax.utils.typing import PRNGKey


class ObjectivesTest(parameterized.TestCase):
  @parameterized.parameters({'key': random.key(0), 'num_index_points': 10})
  def test_negative_log_likelihood(self, key: PRNGKey, num_index_points: int):
    key1, key2, key3 = random.split(key, 3)

    mean = random.uniform(key1, (num_index_points,))
    cov = random.uniform(key2, (num_index_points,)) * jnp.identity(
      num_index_points
    )
    prediction = multivariate_normal.multivariate_normal(mean, cov)

    base_samples = random.normal(key3, (num_index_points,))
    targets = multivariate_normal.sample(prediction, base_samples)

    result = objectives.negative_log_likelihood(multivariate_normal.logpdf)(
      prediction,
      targets,
    )

    expected = multivariate_normal.logpdf(
      prediction,
      targets,
    )

    assert_trees_all_close(result, -expected, atol=1e-4)

  @parameterized.parameters({'key': random.key(0), 'num_index_points': 10})
  def test_penalized(self, key: PRNGKey, num_index_points: int):
    key1, key2, key3, key4 = random.split(key, 4)

    mean = random.uniform(key1, (num_index_points,))
    cov = random.uniform(key2, (num_index_points,)) * jnp.identity(
      num_index_points
    )
    prediction = multivariate_normal.multivariate_normal(mean, cov)

    base_samples = random.normal(key3, (num_index_points,))
    targets = multivariate_normal.sample(prediction, base_samples)

    penalization = random.uniform(key4)

    result = objectives.transformations.penalized(
      objectives.negative_log_likelihood(multivariate_normal.logpdf),
      penalization,
    )(
      prediction,
      targets,
    )

    expected = multivariate_normal.logpdf(
      prediction,
      targets,
    )

    assert_trees_all_close(result, -expected + penalization, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
