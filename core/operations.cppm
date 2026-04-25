export module pml:operations;

import std;
import :tensor;
import :iterator;

export namespace pml {

// Elementwise operation template
template <typename T, typename Op, typename LoopOp>
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

template <typename T> inline T add_operation(const T& left, const T& right) {
    return left + right;
}

template <typename T>
inline void add_operation_loop(const T* left, const T* right, T* result, const std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        result[i] = left[i] + right[i];
    }
}

template <typename T> Tensor<T> add(const Tensor<T>& left, const Tensor<T>& right) {
    return elementwise_operation(left, right, add_operation<T>, add_operation_loop<T>);
}

template <typename T, typename Op>
inline Tensor<T> reduction_operation(const Tensor<T>& operand, const std::size_t axis,
                                     const bool keep_dims, Op op) {
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

    op(iterator_result, iterator_operand, shape_at_axis);

    return result;
}

template <typename T>
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

template <typename T>
Tensor<T> sum(const Tensor<T>& operand, const std::size_t axis, const bool keep_dims = false) {
    return reduction_operation(operand, axis, keep_dims, reduction_operation_sum<T>);
}

template <typename T, typename Op>
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

template <typename T> Tensor<T> as_contiguous_tensor(const Tensor<T>& operand) {
    if (operand.is_contiguous()) {
        // FIXME: Return the same tensor
        throw std::invalid_argument("Tensor must have non-contiguous memory.");
    }

    Tensor<T> result(operand.get_shape());

    TensorIterator<T> iterator_result(result.get_data(), result.get_shape(), result.get_strides());

    TensorIterator<T> iterator_operand(operand.get_data(), operand.get_shape(),
                                       operand.get_strides());

    while (auto* res = iterator_result.next()) {
        *res = *iterator_operand.next();
    }

    return result;
}

} // namespace pml
