from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.core.prediction import models
from boax.core.prediction.models import kernels, likelihoods, means
from boax.utils.functools import const
from boax.utils.typing import Array, Numeric, PRNGKey


class ModelTransformationTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': jnp.array(0.2),
      'noise': 1e-4,
      'bounds': jnp.array([[-1, 1]]),
      'num_index_points': 10,
    }
  )
  def test_transformed(
    self,
    key: PRNGKey,
    length_scale: Numeric,
    noise: float,
    bounds: Array,
    num_index_points: int,
  ):
    key1, key2 = random.split(key)

    index_points = random.uniform(
      key1,
      shape=(num_index_points, 1),
      minval=bounds[:, 0],
      maxval=bounds[:, 1],
    )
    loc, scale = random.uniform(key2, shape=(2, 1))

    model = models.gaussian_process.exact(
      means.zero(),
      kernels.rbf(length_scale),
      likelihoods.gaussian(noise),
    )

    transformed = models.transformations.transformed(
      model,
      input_transformation_fn=models.transformations.input.normalized(bounds),
      outcome_transformation_fn=models.transformations.utils.chained(
        distributions.transformations.mvn_to_norm,
        models.transformations.outcome.scaled(
          loc, scale, distributions.normal.scale
        ),
      ),
    )

    results = transformed(index_points)
    expected = distributions.normal.scale(
      distributions.transformations.mvn_to_norm(
        model(models.transformations.input.normalized(bounds)(index_points)),
      ),
      loc=loc,
      scale=scale,
    )

    assert_trees_all_close(results.loc, expected.loc, atol=1e-4)
    assert_trees_all_close(results.scale, expected.scale, atol=1e-4)

  @parameterized.parameters({'key': random.key(0), 'num_index_points': 10})
  def test_joined(self, key: PRNGKey, num_index_points: int):
    samples = random.uniform(key, shape=(4, num_index_points))
    preds1 = distributions.normal.normal(samples[0], samples[1])
    preds2 = distributions.normal.normal(samples[2], samples[3])

    model = models.transformations.joined(const(preds1), const(preds2))

    result1, result2 = model(jnp.empty((num_index_points,)))

    assert_trees_all_close(result1.loc, preds1.loc, atol=1e-4)
    assert_trees_all_close(result1.scale, preds1.scale, atol=1e-4)
    assert_trees_all_close(result2.loc, preds2.loc, atol=1e-4)
    assert_trees_all_close(result2.scale, preds2.scale, atol=1e-4)

  @parameterized.parameters(
    {'key': random.key(0), 'num_index_points': 10, 'num_samples': 5}
  )
  def test_sampled(self, key: PRNGKey, num_index_points: int, num_samples: int):
    key1, key2 = random.split(key)

    preds = distributions.normal.normal(
      *random.uniform(key1, shape=(2, num_index_points))
    )
    samples = random.normal(key2, shape=(num_samples, num_index_points))

    model = models.transformations.sampled(
      const(preds),
      lambda _, s: s,
      samples,
    )

    result = model(jnp.empty((num_index_points,)))

    assert_trees_all_close(result, samples, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
