from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax import acquisitions
from boax.acquisitions import surrogates
from boax.core import samplers
from boax.utils.typing import Array, PRNGKey


class AcquisitionsTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'num_observations': 10,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_probability_of_improvement(
    self,
    key: PRNGKey,
    bounds: Array,
    num_observations: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2 = random.split(key)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.probability_of_improvement(
      bounds, 1.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key2, posterior, observation_index_points, observations
    )

    assert_shape(batch, (1, bounds.shape[0]))
    assert_shape(value, ())

  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'num_observations': 10,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_log_probability_of_improvement(
    self,
    key: PRNGKey,
    bounds: Array,
    num_observations: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2 = random.split(key)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.log_probability_of_improvement(
      bounds, 1.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key2, posterior, observation_index_points, observations
    )

    assert_shape(batch, (1, bounds.shape[0]))
    assert_shape(value, ())

  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'num_observations': 10,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_expected_improvement(
    self,
    key: PRNGKey,
    bounds: Array,
    num_observations: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2 = random.split(key)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.expected_improvement(
      bounds, 1.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key2, posterior, observation_index_points, observations
    )

    assert_shape(batch, (1, bounds.shape[0]))
    assert_shape(value, ())

  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'num_observations': 10,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_log_expected_improvement(
    self,
    key: PRNGKey,
    bounds: Array,
    num_observations: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2 = random.split(key)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.log_expected_improvement(
      bounds, 1.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key2, posterior, observation_index_points, observations
    )

    assert_shape(batch, (1, bounds.shape[0]))
    assert_shape(value, ())

  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'num_observations': 10,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_upper_confidence_bound(
    self,
    key: PRNGKey,
    bounds: Array,
    num_observations: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2 = random.split(key)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.upper_confidence_bound(
      bounds, 2.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key2, posterior, observation_index_points, observations
    )

    assert_shape(batch, (1, bounds.shape[0]))
    assert_shape(value, ())

  @parameterized.parameters(
    {
      'key': random.key(0),
      'bounds': jnp.array([[-1, 1]]),
      'batch_size': 3,
      'num_observations': 10,
      'num_samples': 32,
      'num_raw_samples': 32,
      'num_restarts': 2,
      'num_steps': 10,
    },
  )
  def test_q_upper_confidence_bound(
    self,
    key: PRNGKey,
    bounds: Array,
    batch_size: int,
    num_observations: int,
    num_samples: int,
    num_raw_samples: int,
    num_restarts: int,
    num_steps: int,
  ):
    key1, key2, key3 = random.split(key, 3)

    observation_index_points = random.uniform(
      key1,
      (num_observations, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    observations = jnp.sin(jnp.pi * observation_index_points[..., 0])

    sampler = samplers.multivariate_normal(key2, (num_samples,))
    surrogate = surrogates.single_task_gaussian_process(
      bounds, 0.01, num_steps, num_raw_samples, num_restarts
    )
    acquisition = acquisitions.q_upper_confidence_bound(
      bounds, batch_size, sampler, (0,), 2.0, num_raw_samples, num_restarts
    )

    params = surrogate.init()
    fitted_params = surrogate.update(
      params, observation_index_points, observations
    )
    posterior = surrogate.posterior(
      fitted_params, observation_index_points, observations
    )

    batch, value = acquisition(
      key3, posterior, observation_index_points, observations
    )

    assert_shape(batch, (batch_size, bounds.shape[0]))
    assert_shape(value, ())


if __name__ == '__main__':
  absltest.main()
