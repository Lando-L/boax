from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core import distributions, models
from boax.core.models import kernels, likelihoods, means
from boax.utils.typing import Array, Numeric, PRNGKey, Shape


class ModelTransformationTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'bounds': jnp.array([[-1, 1]]),
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'bounds': jnp.array([[-1, 1]] * 3),
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'bounds': jnp.array([[-1, 1]] * 2),
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_transformed(
    self,
    key: PRNGKey,
    length_scale: Numeric,
    noise: float,
    bounds: Array,
    b: Shape,
    n: int,
    d: int,
  ):
    key1, key2 = random.split(key)

    index_points = random.uniform(
      key1, shape=b + (n, d), minval=bounds[:, 0] * 5, maxval=bounds[:, 1] * 5
    )

    model = models.gaussian_process.exact(
      means.zero(),
      kernels.rbf(length_scale),
      likelihoods.gaussian(noise),
    )

    loc, scale = random.uniform(key2, shape=(2, 1))

    transformed = models.transformations.transformed(
      model,
      input_transformation_fn=models.transformations.input.normalized(bounds),
      outcome_transformation_fn=models.transformations.outcome.scaled(
        loc, scale, distributions.multivariate_normal.scale
      ),
    )

    results = transformed(index_points)

    expected = distributions.multivariate_normal.scale(
      model(models.transformations.input.normalized(bounds)(index_points)),
      loc,
      scale,
    )

    assert_trees_all_close(results.mean, expected.mean, atol=1e-4)
    assert_trees_all_close(results.cov, expected.cov, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
