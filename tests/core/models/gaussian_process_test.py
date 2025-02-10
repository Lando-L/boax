from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core import models
from boax.core.models import kernels, likelihoods, means
from boax.utils.typing import Numeric, PRNGKey, Shape


class GaussianProcessesTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'num_observation_points': 5,
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'num_observation_points': 5,
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'num_observation_points': 5,
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_exact_gaussian_process(
    self,
    key: PRNGKey,
    length_scale: Numeric,
    noise: float,
    num_observation_points: int,
    b: Shape,
    n: int,
    d: int,
  ):
    key1, key2 = random.split(key)

    index_points = random.uniform(key1, shape=b + (n, d), minval=-1, maxval=1)

    model = models.gaussian_process.exact(
      means.zero(),
      kernels.rbf(length_scale),
      likelihoods.gaussian(noise),
    )

    mean, cov = model(index_points)

    assert_shape(mean, b + (n,))
    assert_shape(cov, b + (n, n))

    observation_index_points = random.uniform(
      key2, shape=(num_observation_points, d), minval=-1, maxval=1
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

    assert_shape(mean, b + (n,))
    assert_shape(cov, b + (n, n))


if __name__ == '__main__':
  absltest.main()
