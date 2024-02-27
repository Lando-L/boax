from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax import optimization
from boax.core import distributions
from boax.optimization import acquisitions, constraints
from boax.prediction import kernels, means, models


class OptimizationConstructionTest(parameterized.TestCase):
  def test_construct(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    acqf = optimization.construct(
      models.outcome_transformed(
        models.gaussian_process(
          means.zero(),
          kernels.rbf(jnp.array(0.2)),
        ),
        distributions.multivariate_normal.as_normal,
      ),
      acquisitions.posterior_mean(),
    )

    mean = acqf(index_points)

    self.assertEqual(mean.shape, (10,))

  def test_construct_constrained(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    acqf = optimization.construct_constrained(
      models.joined(
        models.outcome_transformed(
          models.gaussian_process(
            means.zero(),
            kernels.rbf(jnp.array(0.2)),
          ),
          distributions.multivariate_normal.as_normal,
        ),
        models.outcome_transformed(
          models.gaussian_process(
            means.zero(),
            kernels.rbf(jnp.array(0.2)),
          ),
          distributions.multivariate_normal.as_normal,
        ),
      ),
      acquisitions.expected_improvement(0.0),
      constraints.less_or_equal(0.0),
    )

    result = acqf(index_points)

    self.assertEqual(result.shape, (10,))

  def test_construct_log_constrained(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)

    acqf = optimization.construct_log_constrained(
      models.joined(
        models.outcome_transformed(
          models.gaussian_process(
            means.zero(),
            kernels.rbf(jnp.array(0.2)),
          ),
          distributions.multivariate_normal.as_normal,
        ),
        models.outcome_transformed(
          models.gaussian_process(
            means.zero(),
            kernels.rbf(jnp.array(0.2)),
          ),
          distributions.multivariate_normal.as_normal,
        ),
      ),
      acquisitions.log_expected_improvement(0.0),
      constraints.log_less_or_equal(0.0),
    )

    result = acqf(index_points)

    self.assertEqual(result.shape, (10,))


if __name__ == '__main__':
  absltest.main()
