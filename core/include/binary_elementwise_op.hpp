#pragma once
#include <concepts>
#include <tensor.hpp>
#include <tensor_iterator.hpp>

namespace pml {

template <typename F, typename T>
concept BinaryElementwiseOp = requires(F f, const T& l, const T& r) {
    { f(l, r) } -> std::convertible_to<T>;
};

template <typename F, typename T>
concept BinaryElementwiseLoopOp = requires(F f, const T* l, const T* r, T* res, std::size_t n) {
    { f(l, r, res, n) } -> std::same_as<void>;
};

// Elementwise operation template
template <Number T, BinaryElementwiseOp<T> Op, BinaryElementwiseLoopOp<T> LoopOp>
inline Tensor<T> elementwise_operation(const Tensor<T>& left, const Tensor<T>& right, Op op,
                                       LoopOp loop_op) {
    if (!are_shapes_broadcastable(left.get_shape(), right.get_shape())) {
        throw std::invalid_argument("Tensors are not broadcastable.");
    }

    auto [result_shape, left_strides, right_strides] = broadcast_shapes_and_strides(
        left.get_shape(), left.get_strides(), right.get_shape(), right.get_strides());

    Tensor<T> result(result_shape);

    TensorIterator<T> left_iter(left.get_data(), result_shape, left_strides);
    TensorIterator<T> right_iter(right.get_data(), result_shape, right_strides);
    TensorIterator<T> result_iter(result.get_data(), result.get_shape(), result.get_strides());

    std::size_t gcd_block_sz =
        std::gcd(left_iter.contiguous_block_size(), right_iter.contiguous_block_size());
    if (left_iter.contiguous_block_size() == right_iter.contiguous_block_size()) {
        std::size_t block_sz = left_iter.contiguous_block_size();
        while (auto* out = result_iter.advance(block_sz)) {
            auto* l = left_iter.next_contiguous_block();
            auto* r = right_iter.next_contiguous_block();

            loop_op(l, r, out, block_sz);
        }
    } else if (gcd_block_sz == std::min(left_iter.contiguous_block_size(),
                                        right_iter.contiguous_block_size()) &&
               std::max(left_iter.contiguous_block_size(), right_iter.contiguous_block_size()) %
                       gcd_block_sz ==
                   0) {
        while (auto* out = result_iter.advance(gcd_block_sz)) {
            auto* l = left_iter.advance(gcd_block_sz);
            auto* r = right_iter.advance(gcd_block_sz);

            loop_op(l, r, out, gcd_block_sz);
        }
    } else {
        while (auto* out = result_iter.next()) {
            auto* l = left_iter.next();
            auto* r = right_iter.next();

            *out = op(*l, *r);
        }
    }

    return result;
}

} // namespace pml
