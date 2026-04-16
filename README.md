# PrimitiveML — Minimal tensor runtime & inference framework in C [LEGACY]

**PrimitiveML** is a minimalist tensor library and inference framework written in C, inspired by PyTorch.  
It provides a low-level tensor runtime and a modular layer system (Module/Linear/Embedding) with `forward()`-style calls similar to PyTorch. The repository includes an example of running a small character-based GPT-2 model exported from PyTorch (see `nanogpt).

## Features

- Low-level tensor runtime with dynamic shapes, dtypes and strides.
- Tensor ops: `reshape`, `transpose`, `unsqueeze`.
- Element-wise ops with automatic broadcasting, reductions and matrix multiplication.
- Basic activations: ReLU, Sigmoid, Softmax.
- Modular API: `Module`, `Linear`, `Embedding` with `forward()` semantics.
- Demo: inference of a small char-based GPT-2 model exported from PyTorch.
- CLI for text generation.

## Requirements

- C compiler (clang or gcc)
- CMake
- Python + PyTorch (if you want to run example notebook)

## Build

From the repository root:

```sh
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      ..
make
```

## Project layout

```
.
├── CMakeLists.txt
├── core/          # tensor implementation
├── include/       # public headers (modules, functional, layers)
├── nanogpt/       # Jupyter notebook and PyTorch model code
├── src/           # C implementation of layers & CLI
└── tests/         # unit tests
```

## Tests

Tests are experimental and incomplete. Some unit tests exist, but coverage is partial and many modules are not yet tested. Use with caution.
