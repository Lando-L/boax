from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.prediction import kernels, likelihoods, means, models


class ProcessesTest(parameterized.TestCase):
  def test_gaussian_process(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.gaussian_process(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_gaussian_process_regression(self):
    key_1, key_2 = random.split(random.key(0))

    index_points = random.uniform(key_1, shape=(10, 1), minval=-1, maxval=1)
    observation_index_points = random.uniform(
      key_2, shape=(5, 1), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.gaussian_process_regression(
      observation_index_points,
      observations,
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_predictive(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.gaussian_process(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
    )

    predictive = models.predictive(model, likelihoods.gaussian(1e-4))

    mean, cov = predictive(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def sampled(self):
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)
    base_samples = random.normal(key2, shape=(10,))

    model = models.gaussian_process(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
    )

    sampled = models.sampled(model, distributions.multivariate_normal.sample)

    samples = sampled(index_points)(base_samples)

    self.assertEqual(samples.shape, (10,))


if __name__ == '__main__':
  absltest.main()
