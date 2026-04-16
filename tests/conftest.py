import ctypes
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class CTensor(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p)]


def find_shared_object() -> pathlib.Path:
    candidates = [
        PROJECT_ROOT / "build/tests/libpmltest.so",
        PROJECT_ROOT / "tests/libpmltest.so",
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c)
        if p.exists():
            return p
    raise RuntimeError("Shared library not found")


@pytest.fixture(scope="session")
def lib():
    so_path = find_shared_object()
    lib = ctypes.CDLL(str(so_path))

    lib.create_tensor_py.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.create_tensor_py.restype = ctypes.POINTER(CTensor)

    lib.destroy_tensor_py.argtypes = [ctypes.POINTER(CTensor)]
    lib.destroy_tensor_py.restype = None

    lib.get_tensor_shape_py.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.get_tensor_shape_py.restype = None

    lib.get_add_operation_result.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.get_add_operation_result.restype = None

    lib.create_tensor_scalar_py.argtypes = [ctypes.c_float]
    lib.create_tensor_scalar_py.restype = ctypes.POINTER(CTensor)

    lib.get_sum_operation_result.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_size_t,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.get_sum_operation_result.restype = None

    return lib
