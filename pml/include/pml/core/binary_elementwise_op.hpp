#pragma once
#include <concepts>
#include <numeric>
#include <pml/core/tensor.hpp>
#include <pml/core/tensor_iterator.hpp>

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
        for (std::size_t i = 0; i < n; i++) {
            result[i] = Op{}(left[i], right);
        }
    }
};

inline std::size_t scalar_broadcast_len(const std::vector<std::size_t>& strides) {
    std::size_t n_broadcasted_dims = 0;
    for (std::size_t i = strides.size(); i-- > 0;) {
        if (strides[i] != 0)
            break;
        n_broadcasted_dims++;
    }
    return n_broadcasted_dims;
}

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
        return result;
    }

    if (gcd_block_sz != 1 &&
        gcd_block_sz ==
            std::min(left_iter.contiguous_block_size(), right_iter.contiguous_block_size()) &&
        std::max(left_iter.contiguous_block_size(), right_iter.contiguous_block_size()) %
                gcd_block_sz ==
            0) {
        while (auto* out = result_iter.advance(gcd_block_sz)) {
            auto* l = left_iter.advance(gcd_block_sz);
            auto* r = right_iter.advance(gcd_block_sz);

            Kernel::vector_vector(l, r, out, gcd_block_sz);
        }
        return result;
    }

    std::size_t left_scalar_suffix_len = scalar_broadcast_len(left_strides);
    std::size_t right_scalar_suffix_len = scalar_broadcast_len(right_strides);
    if (gcd_block_sz == 1 && ((left_scalar_suffix_len > 0 && right_scalar_suffix_len == 0 &&
                               right_iter.contiguous_block_size() > 1) ||
                              (left_scalar_suffix_len == 0 && right_scalar_suffix_len > 0 &&
                               left_iter.contiguous_block_size() > 1))) {
        if (right_scalar_suffix_len == 0) {

            std::size_t logic_block_sz =
                std::accumulate(result_shape.end() - left_scalar_suffix_len, result_shape.end(), 1,
                                std::multiplies<std::size_t>());
            std::size_t contiguous_vec_len =
                std::gcd(logic_block_sz, right_iter.contiguous_block_size());

            while (auto* l = left_iter.advance(logic_block_sz)) {

                for (std::size_t i = 0; i < logic_block_sz / contiguous_vec_len; i++) {
                    auto* r = right_iter.advance(contiguous_vec_len);
                    auto* out = result_iter.advance(contiguous_vec_len);

                    Kernel::scalar_vector(*l, r, out, contiguous_vec_len);
                }
            }
        } else {

            std::size_t logic_block_sz =
                std::accumulate(result_shape.end() - right_scalar_suffix_len, result_shape.end(), 1,
                                std::multiplies<std::size_t>());
            std::size_t contiguous_vec_len =
                std::gcd(logic_block_sz, left_iter.contiguous_block_size());
            while (auto* r = right_iter.advance(logic_block_sz)) {
                for (std::size_t i = 0; i < logic_block_sz / contiguous_vec_len; i++) {
                    auto* l = left_iter.advance(contiguous_vec_len);
                    auto* out = result_iter.advance(contiguous_vec_len);

                    Kernel::vector_scalar(l, *r, out, contiguous_vec_len);
                }
            }
        }
        return result;
    }

    while (auto* out = result_iter.next()) {
        auto* l = left_iter.next();
        auto* r = right_iter.next();

        *out = Kernel::scalar_scalar(*l, *r);
    }

    return result;
}

} // namespace pml
