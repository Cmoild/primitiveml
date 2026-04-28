import ctypes
import numpy as np
import numpy.testing as npt


def test_exp(
    lib: ctypes.CDLL,
):
    rng = np.random.default_rng(42)

    for _ in range(100):
        expected = rng.normal(size=(3005,)).astype(np.float32)
        out = np.copy(expected)

        expected = np.exp(expected)

        lib.test_fast_exp(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(out.size),
        )

        npt.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

