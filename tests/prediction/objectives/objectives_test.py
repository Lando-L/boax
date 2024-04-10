from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.prediction import objectives


class ObjectivesTest(parameterized.TestCase):
  def test_negative_log_likelihood(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    mean = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    prediction = multivariate_normal.multivariate_normal(mean, cov)

    base_samples = random.normal(key3, (10,))
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

  def test_penalized(self):
    key1, key2, key3, key4 = random.split(random.key(0), 4)

    mean = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    prediction = multivariate_normal.multivariate_normal(mean, cov)

    base_samples = random.normal(key3, (10,))
    targets = multivariate_normal.sample(prediction, base_samples)

    penalization = random.uniform(key4)

    result = objectives.penalized(
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
