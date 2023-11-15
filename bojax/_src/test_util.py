import numpy as np
from absl.testing import parameterized
from jax import dtypes, random

SEED = 42


class BojaxtestCase(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.rng = random.key(SEED)

    def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
        if canonicalize_dtypes:
            self.assertEqual(
                dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True),
                dtypes.canonicalize_dtype(y.dtype, allow_extended_dtype=True),
            )
        else:
            self.assertEqual(x.dtype, y.dtype)

    def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg=""):
        if check_dtypes:
            self.assertDtypesMatch(x, y)

        with np.errstate(over="ignore"):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=err_msg)

    def assertArrayAllClose(self, x, y, *, check_dtypes=True, err_msg=""):
        if check_dtypes:
            self.assertDtypesMatch(x, y)

        self.assertEqual(x.shape, y.shape)

        with np.errstate(over="ignore"):
            np.testing.assert_allclose(x, y, err_msg=err_msg)
