#pragma once
#include <concepts>
#include <pml/core/tensor.hpp>
#include <pml/core/tensor_iterator.hpp>

namespace pml {

template <typename F, typename T>
concept UnaryElementwiseKernel = requires(F f, const T* o, T* res, const std::size_t n) {
    { f(o, res, n) } -> std::same_as<void>;
};

template <Number T, UnaryElementwiseKernel<T> Kernel>
Tensor<T> elementwise_unary_operation(const Tensor<T>& operand, Kernel kernel) {
    Tensor<T> result(operand.get_shape());

    TensorIterator<T> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), operand.get_shape(),
                                       operand.get_strides());

    std::size_t block_sz = iterator_operand.contiguous_block_size();
    while (auto* res = iterator_result.advance(block_sz)) {
        auto* o = iterator_operand.next_contiguous_block();

        kernel(o, res, block_sz);
    }

    return result;
}

} // namespace pml
