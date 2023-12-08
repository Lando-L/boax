import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization.acquisitions import (
  log_expected_improvement,
  log_probability_of_improvement,
  upper_confidence_bound,
)
from boax.util import const


class AcquisitionsTest(parameterized.TestCase):
  def test_upper_confidence_bound(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    candidates = jnp.empty((1, 1))

    ucb = upper_confidence_bound(beta=1.0, process=const((loc, cov)))(
      candidates
    )
    expected = loc + jnp.sqrt(jnp.diag(cov))

    np.testing.assert_allclose(ucb, expected, atol=1e-4)

  def test_log_expected_improvement(self):
    loc = jnp.array(-0.5)
    cov = jnp.eye(1)
    candidates = jnp.empty((1, 1))

    lei = log_expected_improvement(best=0.0, process=const((loc, cov)))(
      candidates
    )
    expected = jnp.log(0.1978)[..., jnp.newaxis]

    np.testing.assert_allclose(lei, expected, atol=1e-4)

  def test_log_probability_of_improvement(self):
    loc = jnp.zeros(())
    cov = jnp.eye(1)
    candidates = jnp.empty((1, 1))

    lpi = log_probability_of_improvement(best=1.96, process=const((loc, cov)))(
      candidates
    )
    expected = jnp.log(0.025)[..., jnp.newaxis]

    np.testing.assert_allclose(lpi, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
