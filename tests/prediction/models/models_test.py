from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.prediction import models
from boax.prediction.models import kernels, likelihoods, means


class ProcessesTest(parameterized.TestCase):
  def test_gaussian_process(self):
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)

    model = models.gaussian_process(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
      likelihoods.gaussian(1e-4),
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

    observation_index_points = random.uniform(
      key2, shape=(5, 1), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.gaussian_process(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
      likelihoods.gaussian(1e-4),
      observation_index_points,
      observations,
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_multi_fidelity_regression(self):
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 2), minval=-1, maxval=1)

    model = models.multi_fidelity(
      means.zero(),
      kernels.linear_truncated(
        kernels.matern_five_halves(jnp.array(0.2)),
        kernels.matern_five_halves(jnp.array(1.4)),
        jnp.array(1.0),
      ),
      likelihoods.gaussian(1e-4),
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

    observation_index_points = random.uniform(
      key2, shape=(5, 2), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.multi_fidelity(
      means.zero(),
      kernels.linear_truncated(
        unbiased=kernels.matern_five_halves(jnp.array(0.2)),
        biased=kernels.matern_five_halves(jnp.array(1.4)),
        power=jnp.array(1.0),
      ),
      likelihoods.gaussian(1e-4),
      observation_index_points,
      observations,
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_scaled(self):
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)
    loc, scale = random.uniform(key2, shape=(2, 1), minval=0, maxval=1)

    model = models.scaled(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
        likelihoods.gaussian(1e-4),
      ),
      distributions.multivariate_normal.scale,
      loc=loc,
      scale=scale,
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_sampled(self):
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)

    model = models.sampled(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
        likelihoods.gaussian(1e-4),
      ),
      distributions.multivariate_normal.sample,
      random.normal(key2, shape=(5, 10)),
    )

    samples = model(index_points)

    self.assertEqual(
      samples.shape,
      (
        5,
        10,
      ),
    )

  def test_joined(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.joined(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
        likelihoods.gaussian(1e-4),
      ),
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.5)),
        likelihoods.gaussian(1e-4),
      ),
    )

    mvn_a, mvn_b = model(index_points)

    self.assertEqual(mvn_a.mean.shape, (10,))
    self.assertEqual(mvn_a.cov.shape, (10, 10))
    self.assertEqual(mvn_b.mean.shape, (10,))
    self.assertEqual(mvn_b.cov.shape, (10, 10))


if __name__ == '__main__':
  absltest.main()
