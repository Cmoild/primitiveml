# PrimitiveML: tensor runtime and inference framework in C++

**PrimitiveML** is a minimalist tensor library and inference framework written in C++, C and x86-64 Assembly, inspired by PyTorch and NumPy.
It provides a low-level tensor runtime and a modular layer system with `forward()`-style calls, similar to PyTorch.

## Features

- Low-level tensor runtime with dynamic shapes and strides.
- Element-wise operations with automatic broadcasting, reductions and matrix multiplication.
- Modular API: `Module`, `Linear` and `forward()` semantics.
- No dependencies: the library implements everything from low-level kernels to high-level abstractions.
- C++23 and C23 standards are used.
- Testing in Python with `NumPy` and in C++ with `Catch2`.

## Requirements

- C++ compiler (Clang or GCC)
- C compiler (Clang or GCC)
- CMake
- Python (if you want to run tests)
- x86-64 CPU with AVX2 and FMA support to run optimized math kernels (Unix-only, since the System V ABI is used)

## Build

From the repository root, compile `src/main.cpp` in debug mode:

```sh
make debug
```

From the repository root, compile `src/main.cpp` in release mode:

```sh
make release
```

From the repository root, compile the shared library for Python tests:

```sh
make test
```

## Project layout

```
.
├── CMakeLists.txt
├── LICENCE.md
├── Makefile
├── README.md
├── build
├── core           # tensor implementation
├── include        # API of ML modules
├── pyrightconfig.json
├── src
└── tests          # C++ and Python tests
```

## Tests

C++ tests are implemented with `Catch2`. To run tests you need to execute the following commands:

```sh
make test
cd build/
ctest
```

Tests in Python are implemented with `pytest` and `NumPy`.
The goal of Python tests is to check correctness of different math operations.
To run Python tests you need to execute the following commands:

```sh
make test
pytest
```
