import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.optimization import acquisitions
from boax.optimization.acquisitions import constraints


class AcquisitionsTest(parameterized.TestCase):
  def test_probability_of_improvement(self):
    key = random.key(0)
    n, q = 10, 1

    best = 0.0
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    pi = acquisitions.probability_of_improvement(best)(preds)
    lpi = acquisitions.log_probability_of_improvement(best)(preds)

    self.assertEqual(pi.shape, (n,))
    self.assertEqual(lpi.shape, (n,))
    np.testing.assert_allclose(jnp.log(pi), lpi, atol=1e-4)

  def test_expected_improvement(self):
    key = random.key(0)
    n, q = 10, 1

    best = 0.0
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    ei = acquisitions.expected_improvement(best)(preds)
    lei = acquisitions.log_expected_improvement(best)(preds)

    self.assertEqual(ei.shape, (n,))
    self.assertEqual(lei.shape, (n,))
    np.testing.assert_allclose(jnp.log(ei), lei, atol=1e-4)

  def test_upper_confidence_bound(self):
    key = random.key(0)
    n, q = 10, 1

    beta = 2.0
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    ucb = acquisitions.upper_confidence_bound(beta)(preds)
    expected = jnp.squeeze(loc + jnp.sqrt(beta) * scale)

    self.assertEqual(ucb.shape, (n,))
    np.testing.assert_allclose(ucb, expected, atol=1e-4)

  def test_posterior(self):
    key = random.key(0)
    n, q = 10, 1

    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    posterior_mean = acquisitions.posterior_mean()(preds)
    posterior_scale = acquisitions.posterior_scale()(preds)

    self.assertEqual(posterior_mean.shape, (n,))
    self.assertEqual(posterior_scale.shape, (n,))

    np.testing.assert_allclose(posterior_mean, jnp.squeeze(loc), atol=1e-4)
    np.testing.assert_allclose(posterior_scale, jnp.squeeze(scale), atol=1e-4)

  def test_q_probability_of_improvement(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    best = 0.0
    preds = random.uniform(key, (n, s, q))

    qpoi = acquisitions.q_probability_of_improvement(best)(preds)

    self.assertEqual(qpoi.shape, (n,))

  def test_q_expected_improvement(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    best = 0.0
    preds = random.uniform(key, (n, s, q))

    qei = acquisitions.q_expected_improvement(best)(preds)

    self.assertEqual(qei.shape, (n,))

  def test_q_upper_confidence_bound(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    beta = 2.0
    preds = random.uniform(key, (n, s, q))

    qucb = acquisitions.q_upper_confidence_bound(beta)(preds)

    self.assertEqual(qucb.shape, (n,))

  def test_q_knowledge_gradient(self):
    key = random.key(0)
    s, n = 32, 10

    best = 0.0
    loc, scale = random.uniform(key, (2, n, s, 1))
    preds = distributions.normal.normal(loc, scale)

    qkg = acquisitions.q_knowledge_gradient(best)(preds)

    self.assertEqual(qkg.shape, (n,))
  
  def test_q_multi_fidelity_knowledge_gradient(self):
    key1, key2 = random.split(random.key(0))
    s, n = 32, 10

    best = 0.0
    cost_fn = lambda a, b: a / b[..., jnp.newaxis]
    loc, scale = random.uniform(key1, (2, n, s, 1))
    preds = distributions.normal.normal(loc, scale)
    costs = random.uniform(key2, (n,))

    qmfkg = acquisitions.q_multi_fidelity_knowledge_gradient(best, cost_fn)((preds, costs))

    self.assertEqual(qmfkg.shape, (n,))

  def test_constrained(self):
    key = random.key(0)
    n, q = 10, 1

    best = 0.0
    preds_params, cost_params = random.uniform(key, (2, 2, n, q))
    preds = distributions.normal.normal(*preds_params)
    costs = distributions.normal.normal(*cost_params)
    model = [preds, costs]

    cei = acquisitions.constrained(
      acquisitions.expected_improvement(best),
      constraints.less_or_equal(1.0),
    )(model)

    clei = acquisitions.log_constrained(
      acquisitions.log_expected_improvement(best),
      constraints.log_less_or_equal(1.0),
    )(model)

    self.assertEqual(cei.shape, (n,))
    self.assertEqual(clei.shape, (n,))
    np.testing.assert_allclose(jnp.log(cei), clei, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
