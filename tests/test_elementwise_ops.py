import ctypes
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "t1_shape,t2_shape",
    [
        ([5, 2, 2], [2, 2]),
        ([5, 2, 2], [1, 2, 2]),
        ([4, 3, 2], [2]),
        ([1, 3, 1], [5, 1, 4]),
        ([3, 1], [1, 4]),
        ([100, 1], [1, 200]),
        ([1, 5, 1], [1, 1, 7]),
        ([], [2, 15, 13]),
        ([], []),
        ([1, 15, 13], []),
        ([1000, 1000], [1000, 1000]),
    ],
)
def test_elementwise_operations_contiguous(
    lib: ctypes.CDLL, t1_shape: list[int], t2_shape: list[int]
):
    rng = np.random.default_rng(42)

    ops = {
        'add': [np.add, lib.get_add_operation_result],
        'subtract': [np.subtract, lib.get_subtract_operation_result],
        'multiply': [np.multiply, lib.get_multiply_operation_result],
        'divide': [np.divide, lib.get_divide_operation_result],
    }

    for _ in range(100):
        choice = rng.choice(['add', 'subtract', 'multiply', 'divide'])
        t1 = rng.normal(size=t1_shape).astype(np.float32)
        t2 = rng.normal(size=t2_shape).astype(np.float32)

        expected = ops[choice][0](t1, t2)

        out = np.empty(shape=expected.shape, dtype=np.float32)

        pml_c_t1_p = None
        pml_c_t2_p = None
        if t1.ndim > 0:
            pml_c_t1_p = lib.create_tensor_py(
                t1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                t1.size,
                (ctypes.c_size_t * t1.ndim)(*t1.shape),
                t1.ndim,
            )
        else:
            pml_c_t1_p = lib.create_tensor_scalar_py(ctypes.c_float(float(t1)))
        if t2.ndim > 0:
            pml_c_t2_p = lib.create_tensor_py(
                t2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                t2.size,
                (ctypes.c_size_t * t2.ndim)(*t2.shape),
                t2.ndim,
            )
        else:
            pml_c_t2_p = lib.create_tensor_scalar_py(ctypes.c_float(float(t2)))
        out_num_elems = ctypes.c_size_t(0)
        ops[choice][1](
            pml_c_t1_p,
            pml_c_t2_p,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_num_elems),
        )

        assert expected.size == out_num_elems.value
        npt.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

        lib.destroy_tensor_py(pml_c_t1_p)
        lib.destroy_tensor_py(pml_c_t2_p)
