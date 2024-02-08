from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random, vmap

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
    key1, key2 = random.split(random.key(0))

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)
    observation_index_points = random.uniform(
      key2, shape=(5, 1), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.gaussian_process_regression(
      means.zero(),
      kernels.rbf(jnp.array(0.2)),
    )

    mean, cov = model(observation_index_points, observations)(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))

  def test_predictive(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.predictive(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
      ),
      likelihoods.gaussian(1e-4)
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
      ),
      distributions.multivariate_normal.sample,
      random.normal(key2, shape=(5, 10)),
    )

    samples = model(index_points)

    self.assertEqual(samples.shape, (5, 10,))
  
  def test_fantasized(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    index_points = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)

    model = models.fantasized(
      models.sampled(
        models.gaussian_process(
          means.zero(),
          kernels.rbf(jnp.array(0.2)),
        ),
        distributions.multivariate_normal.sample,
        random.normal(key2, shape=(32, 3)),
      ),
      models.gaussian_process_fantasy(
        means.zero(),
        kernels.rbf(jnp.array(0.5))
      ),
      random.uniform(key3, shape=(3, 1), minval=-1, maxval=1)
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (32, 10))
    self.assertEqual(cov.shape, (32, 10, 10))
  
  def test_joined(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.joined(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
      ),
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.5))
      )
    )

    (a_mean, a_cov), (b_mean, b_cov) = model(index_points)

    self.assertEqual(a_mean.shape, (10,))
    self.assertEqual(a_cov.shape, (10, 10,))
    self.assertEqual(b_mean.shape, (10,))
    self.assertEqual(b_cov.shape, (10, 10,))


  def test_input_transformed(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.input_transformed(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
      ),
      lambda x: x * 2 + 1,
    )

    mean, cov = model(index_points)

    self.assertEqual(mean.shape, (10,))
    self.assertEqual(cov.shape, (10, 10))
  
  def test_outcome_transformed(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    model = models.outcome_transformed(
      models.gaussian_process(
        means.zero(),
        kernels.rbf(jnp.array(0.2)),
      ),
      distributions.multivariate_normal.as_normal,
    )

    loc, scale = model(index_points)

    self.assertEqual(loc.shape, (10,))
    self.assertEqual(scale.shape, (10,))


if __name__ == '__main__':
  absltest.main()
