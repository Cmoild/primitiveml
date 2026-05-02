#pragma once

#include <cstddef>
#include <cmath>
#include <concepts>
#include <tensor.hpp>
#include <tensor_iterator.hpp>
#include <binary_elementwise_op.hpp>
#include <reduction_op.hpp>
#include <unary_elementwise_op.hpp>
#include <cpu_flags.h>
#include <math_kernels.h>

namespace pml {

struct AddOp {
    template <Number T> constexpr T operator()(T a, T b) const {
        return a + b;
    }
};

struct SubtractOp {
    template <Number T> constexpr T operator()(T a, T b) const {
        return a - b;
    }
};

struct MultiplyOp {
    template <Number T> constexpr T operator()(T a, T b) const {
        return a * b;
    }
};

struct DivideOp {
    template <Number T> constexpr T operator()(T a, T b) const {
        return a / b;
    }
};

template <Number T> Tensor<T> add(const Tensor<T>& left, const Tensor<T>& right) {
    using AddKernel = BinaryElementwiseKernelBase<T, AddOp>;
    return elementwise_operation<T, AddKernel>(left, right);
}

template <Number T> Tensor<T> operator+(const Tensor<T>& left, const Tensor<T>& right) {
    return add(left, right);
}

template <Number T> Tensor<T> subtract(const Tensor<T>& left, const Tensor<T>& right) {
    using SubtractKernel = BinaryElementwiseKernelBase<T, SubtractOp>;
    return elementwise_operation<T, SubtractKernel>(left, right);
}

template <Number T> Tensor<T> operator-(const Tensor<T>& left, const Tensor<T>& right) {
    return subtract(left, right);
}

template <Number T> Tensor<T> multiply(const Tensor<T>& left, const Tensor<T>& right) {
    using MultiplyKernel = BinaryElementwiseKernelBase<T, MultiplyOp>;
    return elementwise_operation<T, MultiplyKernel>(left, right);
}

template <Number T> Tensor<T> operator*(const Tensor<T>& left, const Tensor<T>& right) {
    return multiply(left, right);
}

template <Number T> Tensor<T> divide(const Tensor<T>& left, const Tensor<T>& right) {
    using DivideKernel = BinaryElementwiseKernelBase<T, DivideOp>;
    return elementwise_operation<T, DivideKernel>(left, right);
}

template <Number T> Tensor<T> operator/(const Tensor<T>& left, const Tensor<T>& right) {
    return divide(left, right);
}

template <Number T>
inline void reduction_operation_sum(TensorIterator<T>& iterator_result,
                                    TensorIterator<T>& iterator_operand,
                                    const std::size_t axis_size, const std::size_t stride) {
    while (auto* res = iterator_result.next()) {
        *res = T{};
        auto* o = iterator_operand.advance(axis_size);
        if (stride == 1) {
            for (std::size_t i = 0; i < axis_size; i++) {
                *res += o[i];
            }
        } else {
            for (std::size_t i = 0; i < axis_size; i++) {
                *res += *(o + stride * i);
            }
        }
    }
}

template <Number T>
Tensor<T> sum(const Tensor<T>& operand, const std::size_t axis, const bool keep_dims = false) {
    return reduction_operation(operand, axis, keep_dims, reduction_operation_sum<T>);
}

template <typename T> void exp_kernel(const T*, T*, std::size_t) = delete;

template <> inline void exp_kernel<float>(const float* o, float* res, std::size_t n) {
#if defined(__x86_64__) && !defined(_WIN64)
    static const bool has_avx2_fma = avx2_fma_supported();

    if (has_avx2_fma) {
        exp_avx2(const_cast<float*>(o), res, n);
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            res[i] = std::expf(o[i]);
        }
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        res[i] = std::expf(o[i]);
    }
#endif
}

template <> inline void exp_kernel<double>(const double* o, double* res, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        res[i] = std::exp(o[i]);
    }
}

template <typename T>
    requires std::same_as<T, float> || std::same_as<T, double>
Tensor<T> exp(const Tensor<T>& operand) {
    return elementwise_unary_operation(operand, exp_kernel<T>);
}

template <Number T> Tensor<T> as_contiguous_tensor(const Tensor<T>& operand) {
    if (operand.is_contiguous()) {
        return Tensor(operand);
    }

    Tensor<T> result(operand.get_shape());

    TensorIterator<T> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), operand.get_shape(),
                                       operand.get_strides());

    std::size_t block_sz = iterator_operand.contiguous_block_size();
    while (auto* res = iterator_result.advance(block_sz)) {
        auto* src = iterator_operand.next_contiguous_block();

        std::copy(src, src + block_sz, res);
    }

    return result;
}

template <Number U, Number T> Tensor<U> cast(const Tensor<T>& operand) {
    Tensor<U> result(operand.get_shape());

    TensorIterator<U> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), operand.get_shape(),
                                       operand.get_strides());

    std::size_t block_sz = iterator_operand.contiguous_block_size();
    while (auto* res = iterator_result.advance(block_sz)) {
        auto* src = iterator_operand.next_contiguous_block();

        for (std::size_t i = 0; i < block_sz; i++) {
            res[i] = static_cast<U>(src[i]);
        }
    }

    return result;
}

} // namespace pml
