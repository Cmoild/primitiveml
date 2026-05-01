#pragma once

#include <concepts>
#include <cstddef>
#include <tensor.hpp>
#include <tensor_iterator.hpp>
#include <binary_elementwise_op.hpp>
#include <reduction_op.hpp>

namespace pml {

template <Number T> inline T add_operation(const T& left, const T& right) {
    return left + right;
}

template <Number T>
inline void add_operation_loop(const T* left, const T* right, T* result, const std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        result[i] = left[i] + right[i];
    }
}

template <Number T> inline T subtract_operation(const T& left, const T& right) {
    return left - right;
}

template <Number T>
inline void subtract_operation_loop(const T* left, const T* right, T* result, const std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        result[i] = left[i] - right[i];
    }
}

template <Number T> Tensor<T> add(const Tensor<T>& left, const Tensor<T>& right) {
    return elementwise_operation(left, right, add_operation<T>, add_operation_loop<T>);
}

template <Number T> Tensor<T> operator+(const Tensor<T>& left, const Tensor<T>& right) {
    return add(left, right);
}

template <Number T>
inline void reduction_operation_sum(TensorIterator<T>& iterator_result,
                                    TensorIterator<T>& iterator_operand,
                                    const std::size_t axis_size) {
    while (auto* res = iterator_result.next()) {
        *res = T{};
        for (std::size_t i = 0; i < axis_size; i++) {
            auto* o = iterator_operand.next();

            *res += *o;
        }
    }
}

template <Number T>
Tensor<T> sum(const Tensor<T>& operand, const std::size_t axis, const bool keep_dims = false) {
    return reduction_operation(operand, axis, keep_dims, reduction_operation_sum<T>);
}

template <typename F, typename T>
concept UnaryElementwiseOp = requires(F f, const T* o, T* res, const std::size_t n) {
    { f(o, res, n) } -> std::same_as<void>;
};

template <Number T, UnaryElementwiseOp<T> Op>
Tensor<T> elementwise_unary_operation(const Tensor<T>& operand, Op op) {
    Tensor<T> result(operand.get_shape());

    TensorIterator<T> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), operand.get_shape(),
                                       operand.get_strides());

    std::size_t block_sz = iterator_operand.contiguous_block_size();
    while (auto* res = iterator_result.advance(block_sz)) {
        auto* o = iterator_operand.next_contiguous_block();

        op(o, res, block_sz);
    }

    return result;
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
