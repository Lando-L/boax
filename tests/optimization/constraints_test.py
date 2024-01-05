import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core.distributions import multivariate_normal
from boax.optimization import constraints
from boax.utils.functools import const


class ConstraintsTest(parameterized.TestCase):
  def test_range(self):
    key1, key2 = random.split(random.key(0))
    n, q, d = 10, 1, 1

    mean = random.uniform(key1, (q,))
    cov = random.uniform(key2, (q,)) * jnp.identity(q)
    model = const(multivariate_normal.multivariate_normal(mean, cov))
    lower = 0.2
    upper = 0.8
    candidates = jnp.empty((n, q, d))

    le = constraints.less_or_equal(model, lower)(candidates)
    lle = constraints.log_less_or_equal(model, lower)(candidates)

    ge = constraints.greater_or_equal(model, upper)(candidates)
    lge = constraints.log_greater_or_equal(model, upper)(candidates)

    np.testing.assert_allclose(jnp.log(le), lle, atol=1e-4)
    np.testing.assert_allclose(jnp.log(ge), lge, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
