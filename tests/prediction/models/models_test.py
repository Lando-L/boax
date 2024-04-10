import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.prediction import models
from boax.prediction.models import kernels, likelihoods, means
from boax.utils.functools import const


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

  def test_multi_fidelity(self):
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

    preds = distributions.normal.normal(*random.uniform(key1, shape=(2, 10)))
    scaled = random.uniform(key2, shape=(2, 1))

    model = models.scaled(
      const(preds),
      distributions.normal.scale,
      loc=scaled[0],
      scale=scaled[1],
    )

    result = model(jnp.empty((10,)))

    expected = distributions.normal.scale(
      preds,
      scaled[0],
      scaled[1],
    )

    np.testing.assert_allclose(result.loc, expected.loc, atol=1e-4)
    np.testing.assert_allclose(result.scale, expected.scale, atol=1e-4)

  def test_sampled(self):
    key1, key2 = random.split(random.key(0))

    preds = distributions.normal.normal(*random.uniform(key1, shape=(2, 10)))
    samples = random.normal(key2, shape=(5, 10))

    model = models.sampled(
      const(preds),
      lambda _, s: s,
      samples,
    )

    result = model(jnp.empty((10,)))

    np.testing.assert_allclose(result, samples, atol=1e-4)

  def test_joined(self):
    key = random.key(0)

    samples = random.uniform(key, shape=(4, 10))
    preds1 = distributions.normal.normal(samples[0], samples[1])
    preds2 = distributions.normal.normal(samples[2], samples[3])

    model = models.joined(const(preds1), const(preds2))

    result1, result2 = model(jnp.empty((10,)))

    np.testing.assert_allclose(result1.loc, preds1.loc, atol=1e-4)
    np.testing.assert_allclose(result1.scale, preds1.scale, atol=1e-4)
    np.testing.assert_allclose(result2.loc, preds2.loc, atol=1e-4)
    np.testing.assert_allclose(result2.scale, preds2.scale, atol=1e-4)
  
  def test_fantazised(self):
    key1, key2 = random.split(random.key(0))

    samples = random.uniform(key1, shape=(10, 3, 1))
    preds = distributions.normal.normal(*random.uniform(key2, shape=(2, 10, 3)))
    
    model = models.fantasized(
      const(samples),
      const(const(preds)),
      jnp.empty((10, 3))
    )

    result = model(jnp.empty((10,)))

    np.testing.assert_allclose(result.loc, preds.loc, atol=1e-4)
    np.testing.assert_allclose(result.scale, preds.scale, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
