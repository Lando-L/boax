from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core.prediction import models
from boax.core.prediction.models import kernels, likelihoods, means
from boax.utils.typing import Numeric, PRNGKey


class GaussianProcessesTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'num_index_points': 10,
      'num_observation_points': 5,
    }
  )
  def test_exact_gaussian_process(
    self,
    key: PRNGKey,
    length_scale: Numeric,
    noise: float,
    num_index_points: int,
    num_observation_points: int,
  ):
    key1, key2 = random.split(key)

    index_points = random.uniform(
      key1, shape=(num_index_points, 1), minval=-1, maxval=1
    )

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
      'key': random.key(0),
      'unbiased': jnp.array(0.2),
      'biased': jnp.array(1.4),
      'power': jnp.array(1.0),
      'noise': 1e-4,
      'num_index_points': 10,
      'num_observation_points': 5,
    }
  )
  def test_multi_fidelity(
    self,
    key: PRNGKey,
    unbiased: Numeric,
    biased: Numeric,
    power: Numeric,
    noise: float,
    num_index_points: int,
    num_observation_points: int,
  ):
    key1, key2 = random.split(key)

    index_points, fidelities = random.uniform(
      key1, shape=(2, num_index_points, 1), minval=-1, maxval=1
    )

    model = models.gaussian_process.multi_fidelity(
      means.zero(),
      kernels.transformations.linear_truncated(
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
      kernels.transformations.linear_truncated(
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


if __name__ == '__main__':
  absltest.main()
