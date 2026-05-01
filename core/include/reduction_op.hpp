#pragma once
#include <concepts>
#include <tensor.hpp>
#include <tensor_iterator.hpp>

namespace pml {

template <typename F, typename T>
concept ReductionKernel =
    requires(F f, TensorIterator<T>& i_r, TensorIterator<T>& i_o, const std::size_t axis_size) {
        { f(i_r, i_o, axis_size) } -> std::same_as<void>;
    };

template <Number T, ReductionKernel<T> Kernel>
inline Tensor<T> reduction_operation(const Tensor<T>& operand, const std::size_t axis,
                                     const bool keep_dims, Kernel kernel) {
    std::vector<std::size_t> transopsed_shape = operand.get_shape();
    std::vector<std::size_t> transopsed_strides = operand.get_strides();
    std::size_t shape_at_axis = 0;

    if (axis >= 0 && axis < transopsed_shape.size() && axis < transopsed_strides.size()) {
        shape_at_axis = transopsed_shape[axis];
        std::size_t stride_at_axis = transopsed_strides[axis];
        transopsed_shape.erase(transopsed_shape.begin() + axis);
        transopsed_strides.erase(transopsed_strides.begin() + axis);
        transopsed_shape.push_back(shape_at_axis);
        transopsed_strides.push_back(stride_at_axis);
    } else {
        throw std::invalid_argument("Axis is out of bounds.");
    }

    std::vector<std::size_t> result_shape = operand.get_shape();
    if (keep_dims) {
        result_shape[axis] = 1;
    } else {
        result_shape.erase(result_shape.begin() + axis);
    }

    Tensor<T> result(result_shape);

    TensorIterator<T> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), transopsed_shape, transopsed_strides);

    kernel(iterator_result, iterator_operand, shape_at_axis);

    return result;
}

} // namespace pml
