import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.optimization import acquisitions
from boax.utils.functools import const, identity


class AcquisitionsTest(parameterized.TestCase):
  def test_expected_improvement(self):
    key1, key2 = random.split(random.key(0))
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    best = 0.0
    candidates = jnp.empty((n, q, d))

    ei = acquisitions.expected_improvement(model, best)(candidates)
    lei = acquisitions.log_expected_improvement(model, best)(candidates)

    self.assertEqual(ei.shape, (n,))
    self.assertEqual(lei.shape, (n,))
    np.testing.assert_allclose(jnp.log(ei), lei, atol=1e-4)

  def test_probability_of_improvement(self):
    key1, key2 = random.split(random.key(0))
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    best = 1.96
    candidates = jnp.empty((n, q, d))

    pi = acquisitions.probability_of_improvement(model, best)(candidates)
    lpi = acquisitions.log_probability_of_improvement(model, best)(candidates)

    self.assertEqual(pi.shape, (n,))
    self.assertEqual(lpi.shape, (n,))
    np.testing.assert_allclose(jnp.log(pi), lpi, atol=1e-4)

  def test_upper_confidence_bound(self):
    key1, key2 = random.split(random.key(0))
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    beta = 1.0
    candidates = jnp.empty((n, q, d))

    ucb = acquisitions.upper_confidence_bound(model, beta)(candidates)
    expected = jnp.repeat(mean + jnp.sqrt(jnp.diag(cov)), n)

    self.assertEqual(ucb.shape, (n,))
    np.testing.assert_allclose(ucb, expected, atol=1e-4)

  def test_posterior(self):
    key1, key2 = random.split(random.key(0))
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    candidates = jnp.empty((n, q, d))

    posterior_mean = acquisitions.posterior_mean(model)(candidates)
    posterior_scale = acquisitions.posterior_scale(model)(candidates)

    self.assertEqual(posterior_mean.shape, (n,))
    self.assertEqual(posterior_scale.shape, (n,))

    np.testing.assert_allclose(posterior_mean, jnp.repeat(mean, n), atol=1e-4)
    np.testing.assert_allclose(posterior_scale, jnp.repeat(jnp.sqrt(jnp.diag(cov)), n), atol=1e-4)

  def test_q_expected_improvement(self):
    key = random.key(0)
    n, q, d = 10, 5, 1

    model = const(identity)
    base_samples = random.uniform(key, (n, q,))
    best = 0.0
    candidates = jnp.empty((n, q, d))

    qei = acquisitions.q_expected_improvement(model, base_samples, best)(
      candidates
    )

    self.assertEqual(qei.shape, (n,))

  def test_q_probability_of_improvement(self):
    key = random.key(0)
    n, q, d = 10, 5, 1

    model = const(identity)
    base_samples = random.uniform(key, (n, q,))
    tau = 1.0
    best = 0.0
    candidates = jnp.empty((n, q, d))

    qei = acquisitions.q_probability_of_improvement(
      model, base_samples, tau, best
    )(candidates)

    self.assertEqual(qei.shape, (n,))

  def test_q_upper_confidence_bound(self):
    key = random.key(0)
    n, q, d = 10, 5, 1

    model = const(identity)
    base_samples = random.uniform(key, (n, q,))
    beta = 2.0
    candidates = jnp.empty((n, q, d))

    qei = acquisitions.q_upper_confidence_bound(model, base_samples, beta)(
      candidates
    )

    self.assertEqual(qei.shape, (n,))

  def test_constrained(self):
    key1, key2, key3, key4 = random.split(random.key(0), 4)
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    c1 = random.uniform(key3, (q,))
    c2 = random.uniform(key4, (q,))
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    best = 0.0
    candidates = jnp.empty((n, q, d))

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

    self.assertEqual(cei.shape, (n,))
    self.assertEqual(clei.shape, (n,))
    np.testing.assert_allclose(jnp.log(cei), clei, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
