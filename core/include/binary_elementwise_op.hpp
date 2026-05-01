#pragma once
#include <concepts>
#include <tensor.hpp>
#include <tensor_iterator.hpp>

namespace pml {

template <typename K, typename T>
concept BinaryElementwiseKernel =
    requires(const T* l, const T* r, T* res, std::size_t n, const T& a, const T& b) {
        { K::scalar_scalar(a, b) } -> std::convertible_to<T>;
        { K::vector_vector(l, r, res, n) } -> std::same_as<void>;
        { K::scalar_vector(a, r, res, n) } -> std::same_as<void>;
        { K::vector_scalar(l, b, res, n) } -> std::same_as<void>;
    };

template <typename Op, typename T>
concept BinaryOp = requires(Op op, T a, T b) {
    { op(a, b) } -> std::convertible_to<T>;
};

template <Number T, BinaryOp<T> Op> class BinaryElementwiseKernelBase {
  protected:
    BinaryElementwiseKernelBase() = default;

  public:
    static inline T scalar_scalar(const T& left, const T& right) {
        return Op{}(left, right);
    }

    static inline void vector_vector(const T* left, const T* right, T* result,
                                     const std::size_t n) {
        for (std::size_t i = 0; i < n; i++) {
            result[i] = Op{}(left[i], right[i]);
        }
    }

    static inline void scalar_vector(const T& left, const T* right, T* result,
                                     const std::size_t n) {
        for (std::size_t i = 0; i < n; i++) {
            result[i] = Op{}(left, right[i]);
        }
    }

    static inline void vector_scalar(const T* left, const T& right, T* result,
                                     const std::size_t n) {
        scalar_vector(right, left, result, n);
    }
};

// Elementwise operation template
template <Number T, BinaryElementwiseKernel<T> Kernel>
inline Tensor<T> elementwise_operation(const Tensor<T>& left, const Tensor<T>& right) {
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

            Kernel::vector_vector(l, r, out, block_sz);
        }
    } else if (gcd_block_sz == std::min(left_iter.contiguous_block_size(),
                                        right_iter.contiguous_block_size()) &&
               std::max(left_iter.contiguous_block_size(), right_iter.contiguous_block_size()) %
                       gcd_block_sz ==
                   0) {
        while (auto* out = result_iter.advance(gcd_block_sz)) {
            auto* l = left_iter.advance(gcd_block_sz);
            auto* r = right_iter.advance(gcd_block_sz);

            Kernel::vector_vector(l, r, out, gcd_block_sz);
        }
    } else {
        while (auto* out = result_iter.next()) {
            auto* l = left_iter.next();
            auto* r = right_iter.next();

            *out = Kernel::scalar_scalar(*l, *r);
        }
    }

    return result;
}

} // namespace pml
