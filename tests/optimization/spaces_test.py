import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from bojax._src.optimization.spaces.alias import continuous


class SpacesTest(parameterized.TestCase):
  def test_continuous(self):
    bounds = jnp.array([[0.0, 1.0], [-1.0, 1.0]])
    space = continuous(bounds)

    self.assertEqual(space.ndims, 2)

    samples = space.sample(num_samples=100)

    self.assertEqual(samples.shape, (100, 2))

    np.testing.assert_array_less(-samples[..., 0], -bounds[0, 0])
    np.testing.assert_array_less(samples[..., 0], bounds[0, 1])
    np.testing.assert_array_less(-samples[..., 1], -bounds[1, 0])
    np.testing.assert_array_less(samples[..., 1], bounds[1, 1])


if __name__ == '__main__':
  absltest.main()
