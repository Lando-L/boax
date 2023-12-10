import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization.acquisitions import (
  expected_improvement,
  log_expected_improvement,
  log_probability_of_improvement,
  posterior_mean,
  posterior_scale,
  probability_of_improvement,
  upper_confidence_bound,
)
from boax.util import const


class AcquisitionsTest(parameterized.TestCase):
  def test_expected_improvement(self):
    loc = jnp.array(-0.5)
    cov = jnp.eye(1)
    candidates = jnp.empty((1, 1))

    ei = expected_improvement(0.0, const((loc, cov)))(candidates)
    lei = log_expected_improvement(0.0, const((loc, cov)))(candidates)

    np.testing.assert_allclose(jnp.log(ei), lei, atol=1e-4)

  def test_probability_of_improvement(self):
    loc = jnp.zeros(())
    cov = jnp.eye(1)
    candidates = jnp.empty((1, 1))

    pi = probability_of_improvement(1.96, const((loc, cov)))(candidates)
    lpi = log_probability_of_improvement(1.96, const((loc, cov)))(candidates)

    np.testing.assert_allclose(jnp.log(pi), lpi, atol=1e-4)

  def test_upper_confidence_bound(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    candidates = jnp.empty((1, 1))

    ucb = upper_confidence_bound(1.0, const((loc, cov)))(candidates)
    expected = loc + jnp.sqrt(jnp.diag(cov))

    np.testing.assert_allclose(ucb, expected, atol=1e-4)

  def test_posterior(self):
    loc = jnp.zeros(())
    cov = jnp.eye(1)
    candidates = jnp.empty((1, 1))

    mean = posterior_mean(const((loc, cov)))(candidates)
    scale = posterior_scale(const((loc, cov)))(candidates)

    np.testing.assert_allclose(mean, loc, atol=1e-4)
    np.testing.assert_allclose(scale, jnp.sqrt(jnp.diag(cov)), atol=1e-4)


if __name__ == '__main__':
  absltest.main()
