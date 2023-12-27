import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import acquisitions
from boax.utils.functools import const


class AcquisitionsTest(parameterized.TestCase):
  def test_expected_improvement(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    best = 0.0
    candidates = jnp.empty(())

    ei = acquisitions.expected_improvement(model, best)(candidates)
    lei = acquisitions.log_expected_improvement(model, best)(candidates)

    np.testing.assert_allclose(jnp.log(ei), lei, atol=1e-4)

  def test_probability_of_improvement(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    best = 1.96
    candidates = jnp.empty(())

    pi = acquisitions.probability_of_improvement(model, best)(candidates)
    lpi = acquisitions.log_probability_of_improvement(model, best)(candidates)

    np.testing.assert_allclose(jnp.log(pi), lpi, atol=1e-4)

  def test_upper_confidence_bound(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    beta = 1.0
    candidates = jnp.empty(())

    ucb = acquisitions.upper_confidence_bound(model, beta)(candidates)
    expected = loc + jnp.sqrt(jnp.diag(cov))

    np.testing.assert_allclose(ucb, expected, atol=1e-4)

  def test_posterior(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    candidates = jnp.empty(())

    mean = acquisitions.posterior_mean(model)(candidates)
    scale = acquisitions.posterior_scale(model)(candidates)

    np.testing.assert_allclose(mean, loc, atol=1e-4)
    np.testing.assert_allclose(scale, jnp.sqrt(jnp.diag(cov)), atol=1e-4)

  def test_q_expected_improvement(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    base_samples = random.normal(key3, (10,))
    best = 0.0
    candidates = jnp.empty(())

    qei = acquisitions.q_expected_improvement(model, base_samples, best)(
      candidates
    )

    self.assertEqual(qei.shape, (10,))

  def test_q_probability_of_improvement(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    base_samples = random.normal(key3, (10,))
    tau = 1.0
    best = 0.0
    candidates = jnp.empty(())

    qei = acquisitions.q_probability_of_improvement(
      model, base_samples, tau, best
    )(candidates)

    self.assertEqual(qei.shape, (10,))

  def test_q_upper_confidence_bound(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    base_samples = random.normal(key3, (10,))
    beta = 2.0
    candidates = jnp.empty(())

    qei = acquisitions.q_upper_confidence_bound(model, base_samples, beta)(
      candidates
    )

    self.assertEqual(qei.shape, (10,))

  def test_constrained(self):
    key1, key2, key3, key4 = random.split(random.key(0), 4)

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    c1 = random.uniform(key3, (10,))
    c2 = random.uniform(key4, (10,))
    model = const((loc, cov))
    best = 0.0
    candidates = jnp.empty(())

    cei = acquisitions.constrained(
      acquisitions.expected_improvement(model, best),
      const(c1),
      const(c2),
    )(candidates)

    clei = acquisitions.log_constrained(
      acquisitions.log_expected_improvement(model, best),
      const(jnp.log(c1)),
      const(jnp.log(c2)),
    )(candidates)

    print(cei)
    print(clei)

    np.testing.assert_allclose(jnp.log(cei), clei, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
