from collections.abc import Sequence

from absl.testing import absltest, parameterized
from chex import assert_shape, assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.prediction import models
from boax.prediction.models import kernels, likelihoods, means
from boax.utils.functools import const
from boax.utils.typing import Array, Numeric, PRNGKey


class GaussianProcessesTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      "key": random.key(0),
      "length_scale": jnp.array(0.2),
      "noise": 1e-4,
      "num_index_points": 10,
      "num_observation_points": 5
    }
  )
  def test_exact_gaussian_process(self, key: PRNGKey, length_scale: Numeric, noise: float, num_index_points: int, num_observation_points: int):
    key1, key2 = random.split(key)

    index_points = random.uniform(key1, shape=(num_index_points, 1), minval=-1, maxval=1)

    model = models.gaussian_process.exact(
      means.zero(),
      kernels.rbf(length_scale),
      likelihoods.gaussian(noise),
    )

    mean, cov = model(index_points)

    assert_shape(mean, (num_index_points,))
    assert_shape(cov, (num_index_points, num_index_points))

    observation_index_points = random.uniform(
      key2, shape=(num_observation_points, 1), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.gaussian_process.exact(
      means.zero(),
      kernels.rbf(length_scale),
      likelihoods.gaussian(noise),
      observation_index_points,
      observations,
    )

    mean, cov = model(index_points)

    assert_shape(mean, (num_index_points,))
    assert_shape(cov, (num_index_points, num_index_points))

  @parameterized.parameters(
    {
      "key": random.key(0),
      "unbiased": jnp.array(0.2),
      "biased": jnp.array(1.4),
      "power": jnp.array(1.0),
      "noise": 1e-4,
      "num_index_points": 10,
      "num_observation_points": 5
    }
  )
  def test_multi_fidelity(self, key: PRNGKey, unbiased: Numeric, biased: Numeric, power: Numeric, noise: float, num_index_points: int, num_observation_points: int):
    key1, key2 = random.split(key)

    index_points, fidelities = random.uniform(
      key1, shape=(2, num_index_points, 1), minval=-1, maxval=1
    )

    model = models.gaussian_process.multi_fidelity(
      means.zero(),
      kernels.linear_truncated(
        kernels.matern_five_halves(unbiased),
        kernels.matern_five_halves(biased),
        power,
      ),
      likelihoods.gaussian(noise),
    )

    mean, cov = model(index_points, fidelities)

    assert_shape(mean, (num_index_points,))
    assert_shape(cov, (num_index_points, num_index_points))

    observation_index_points, observation_fidelities = random.uniform(
      key2, shape=(2, num_observation_points, 1), minval=-1, maxval=1
    )
    observations = jnp.sin(observation_index_points[..., 0]) + jnp.cos(
      observation_index_points[..., 0]
    )

    model = models.gaussian_process.multi_fidelity(
      means.zero(),
      kernels.linear_truncated(
        kernels.matern_five_halves(unbiased),
        kernels.matern_five_halves(biased),
        power,
      ),
      likelihoods.gaussian(noise),
      observation_index_points,
      observation_fidelities,
      observations,
    )

    mean, cov = model(index_points, fidelities)

    assert_shape(mean, (num_index_points,))
    assert_shape(cov, (num_index_points, num_index_points))

  @parameterized.parameters(
    {"key": random.key(0), "sample_shape": (4, 10), "num_index_points": 10}
  )
  def test_joined(self, key: PRNGKey, sample_shape: Sequence[int], num_index_points: int):
    key = random.key(0)

    samples = random.uniform(key, shape=sample_shape)
    preds1 = distributions.normal.normal(samples[0], samples[1])
    preds2 = distributions.normal.normal(samples[2], samples[3])

    model = models.joined(const(preds1), const(preds2))

    result1, result2 = model(jnp.empty((num_index_points,)))

    assert_trees_all_close(result1.loc, preds1.loc, atol=1e-4)
    assert_trees_all_close(result1.scale, preds1.scale, atol=1e-4)
    assert_trees_all_close(result2.loc, preds2.loc, atol=1e-4)
    assert_trees_all_close(result2.scale, preds2.scale, atol=1e-4)

  @parameterized.parameters(
    {"key": random.key(0), "num_index_points": 10}
  )
  def test_scaled(self, key: PRNGKey, num_index_points: int):
    key1, key2 = random.split(key)

    preds = distributions.normal.normal(*random.uniform(key1, shape=(2, num_index_points)))
    scaled = random.uniform(key2, shape=(2, 1))

    model = models.scaled(
      const(preds),
      distributions.normal.scale,
      loc=scaled[0],
      scale=scaled[1],
    )

    result = model(jnp.empty((num_index_points,)))

    expected = distributions.normal.scale(
      preds,
      scaled[0],
      scaled[1],
    )

    assert_trees_all_close(result.loc, expected.loc, atol=1e-4)
    assert_trees_all_close(result.scale, expected.scale, atol=1e-4)

  @parameterized.parameters(
    {"key": random.key(0), "num_index_points": 10, "num_samples": 5}
  )
  def test_sampled(self, key: PRNGKey, num_index_points: int, num_samples: int):
    key1, key2 = random.split(key)

    preds = distributions.normal.normal(*random.uniform(key1, shape=(2, num_index_points)))
    samples = random.normal(key2, shape=(num_samples, num_index_points))

    model = models.sampled(
      const(preds),
      lambda _, s: s,
      samples,
    )

    result = model(jnp.empty((num_index_points,)))

    assert_trees_all_close(result, samples, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
