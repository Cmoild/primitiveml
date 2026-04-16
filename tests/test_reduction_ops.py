import ctypes
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "shape", [([2, 3, 3]), ([50, 50]), ([1000]), ([5, 5, 5, 5, 5])]
)
def test_sum_reduction_operation_contiguous(lib: ctypes.CDLL, shape: list[int]):
    rng = np.random.default_rng(42)

    for _ in range(100):
        axis = int(rng.integers(0, len(shape)))
        keep_dims = bool(rng.choice([True, False]))

        initial = rng.normal(size=shape).astype(np.float32)
        expected = initial.sum(axis=axis, keepdims=keep_dims)

        out = np.empty(shape=expected.shape, dtype=np.float32)

        pml_c_operand_p = lib.create_tensor_py(
            initial.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            initial.size,
            (ctypes.c_size_t * initial.ndim)(*initial.shape),
            initial.ndim,
        )

        out_num_elems = ctypes.c_size_t(0)
        lib.get_sum_operation_result(
            pml_c_operand_p,
            axis,
            keep_dims,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_num_elems),
        )

        assert expected.size == out_num_elems.value
        npt.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)

        lib.destroy_tensor_py(pml_c_operand_p)
