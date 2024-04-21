from absl.testing import absltest, parameterized
from chex import assert_shape, assert_trees_all_close
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

    poi = acquisitions.probability_of_improvement(best)(preds)
    lpoi = acquisitions.log_probability_of_improvement(best)(preds)

    assert_shape(poi, (n,))
    assert_shape(lpoi, (n,))
    assert_trees_all_close(jnp.log(poi), lpoi, atol=1e-4)

  def test_expected_improvement(self):
    key = random.key(0)
    n, q = 10, 1

    best = 0.0
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    ei = acquisitions.expected_improvement(best)(preds)
    lei = acquisitions.log_expected_improvement(best)(preds)

    assert_shape(ei, (n,))
    assert_shape(lei, (n,))
    assert_trees_all_close(jnp.log(ei), lei, atol=1e-4)

  def test_upper_confidence_bound(self):
    key = random.key(0)
    n, q = 10, 1

    beta = 2.0
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    ucb = acquisitions.upper_confidence_bound(beta)(preds)
    expected = jnp.squeeze(loc + jnp.sqrt(beta) * scale)

    assert_shape(ucb, (n,))
    assert_trees_all_close(ucb, expected, atol=1e-4)

  def test_posterior(self):
    key = random.key(0)
    n, q = 10, 1

    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    posterior_mean = acquisitions.posterior_mean()(preds)
    posterior_scale = acquisitions.posterior_scale()(preds)

    assert_shape(posterior_mean, (n,))
    assert_shape(posterior_scale, (n,))
    assert_trees_all_close(posterior_mean, jnp.squeeze(loc), atol=1e-4)
    assert_trees_all_close(posterior_scale, jnp.squeeze(scale), atol=1e-4)

  def test_q_probability_of_improvement(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    best = 0.0
    preds = random.uniform(key, (s, n, q))

    qpoi = acquisitions.q_probability_of_improvement(best)(preds)

    assert_shape(qpoi, (n,))

  def test_q_expected_improvement(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    best = 0.0
    preds = random.uniform(key, (s, n, q))

    qei = acquisitions.q_expected_improvement(best)(preds)

    assert_shape(qei, (n,))

  def test_q_upper_confidence_bound(self):
    key = random.key(0)
    s, n, q = 32, 10, 5

    beta = 2.0
    preds = random.uniform(key, (s, n, q))

    qucb = acquisitions.q_upper_confidence_bound(beta)(preds)

    assert_shape(qucb, (n,))

  def test_q_knowledge_gradient(self):
    key = random.key(0)
    s, n = 32, 10

    best = 0.0
    loc, scale = random.uniform(key, (2, s, n, 1))
    preds = distributions.normal.normal(loc, scale)

    qkg = acquisitions.q_knowledge_gradient(best)(preds)

    assert_shape(qkg, (n,))

  def test_q_multi_fidelity_knowledge_gradient(self):
    key1, key2 = random.split(random.key(0))
    s, n = 32, 10

    best = 0.0
    cost_fn = lambda a, b: a / b[..., jnp.newaxis]
    loc, scale = random.uniform(key1, (2, s, n, 1))
    preds = distributions.normal.normal(loc, scale)
    costs = random.uniform(key2, (s,))

    qmfkg = acquisitions.q_multi_fidelity_knowledge_gradient(best, cost_fn)(
      (preds, costs)
    )

    assert_shape(qmfkg, (n,))

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

    assert_shape(cei, (n,))
    assert_shape(clei, (n,))
    assert_trees_all_close(jnp.log(cei), clei, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
