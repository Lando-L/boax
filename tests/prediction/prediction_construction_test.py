from absl.testing import absltest, parameterized
from jax import nn, random
from jax import numpy as jnp

from boax import prediction
from boax.core import distributions
from boax.prediction import kernels, likelihoods, means, models, objectives


class PredictionConstructionTest(parameterized.TestCase):
  def test_construct(self):
    key = random.key(0)

    index_points = random.uniform(key, shape=(10, 1), minval=-1, maxval=1)
    observations = jnp.sin(2 * index_points[..., 0])

    params = {
      'amplitude': jnp.zeros(()),
      'length_scale': jnp.zeros(()),
      'noise': jnp.zeros(()),
    }

    def model(params):
      return models.outcome_transformed(
        models.gaussian_process(
          means.zero(),
          kernels.scaled(
            kernels.rbf(params['amplitude']),
            params['length_scale'],
          ),
        ),
        likelihoods.gaussian(params['noise']),
      )

    def objective(params):
      return objectives.negative_log_likelihood(
        distributions.multivariate_normal.logpdf
      )

    def projection(params):
      return {
        'amplitude': nn.softplus(params['amplitude']),
        'length_scale': nn.softplus(params['length_scale']),
        'noise': nn.softplus(params['noise']) + 1e-4,
      }

    loss_fn = prediction.construct(model, objective, projection)
    loss = loss_fn(params, index_points, observations)

    self.assertEqual(loss.shape, ())


if __name__ == '__main__':
  absltest.main()
