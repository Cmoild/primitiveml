#pragma once

#include <cstddef>
#include <stdexcept>
#include <tensor.hpp>
#include <tensor_iterator.hpp>
#include <math_kernels.h>
#include <cpu_flags.h>

namespace pml {

template <Number T>
void matmul_kernel(const T* left, const T* right, T* res, std::size_t M, std::size_t N,
                   std::size_t K) {
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            res[N * i + j] = T{};
            for (std::size_t k = 0; k < K; k++) {
                res[N * i + j] += left[K * i + k] * right[N * k + j];
            }
        }
    }
}

template <typename T> struct MatmulBackend {
    static void run(const T* l, const T* r, T* out, std::size_t M, std::size_t N, std::size_t K) {
        matmul_kernel(l, r, out, M, N, K);
    }
};

template <> struct MatmulBackend<float> {
    static void run(const float* l, const float* r, float* out, std::size_t M, std::size_t N,
                    std::size_t K) {
#if defined(__x86_64__) && !defined(_WIN64)
        static const bool has_avx2_fma = avx2_fma_supported();
        if (has_avx2_fma) {
            gemm(l, r, out, M, N, K, false, false, 1.f, 0.f, K, N, N);
        } else {
            matmul_kernel(l, r, out, M, N, K);
        }
#else
        matmul_kernel(l, r, out, M, N, K);
#endif
    }
};

template <Number T> Tensor<T> matmul(const Tensor<T>& left, const Tensor<T>& right) {
    if (left.ndim() == 0 || right.ndim() == 0) {
        throw std::invalid_argument("Tensor shapes are incompatible with matmul op.");
    }

    if (left.ndim() <= 2 && right.ndim() <= 2) {
        bool left_reshaped = false;
        bool right_reshaped = false;

        std::vector<std::size_t> shape_left(2);

        if (left.ndim() == 1) {
            left_reshaped = true;
            shape_left[0] = 1;
            shape_left[1] = left.get_shape()[0];
        } else {
            shape_left = left.get_shape();
        }

        std::vector<std::size_t> shape_right(2);

        if (right.ndim() == 1) {
            right_reshaped = true;
            shape_right[0] = right.get_shape()[0];
            shape_right[1] = 1;
        } else {
            shape_right = right.get_shape();
        }

        if (shape_left[1] != shape_right[0]) {
            throw std::invalid_argument("Incompatible matrix shapes.");
        }

        Tensor<T> l_contiguous = (left.is_contiguous()) ? left : as_contiguous_tensor(left);
        Tensor<T> r_contiguous = (right.is_contiguous()) ? right : as_contiguous_tensor(right);

        std::vector<std::size_t> result_shape{};
        if (!left_reshaped && !right_reshaped) {
            result_shape = {shape_left[0], shape_right[1]};
        } else if (left_reshaped != right_reshaped) {
            if (left_reshaped) {
                result_shape = {shape_right[1]};
            } else {
                result_shape = {shape_left[0]};
            }
        }
        Tensor<T> result(result_shape);

        MatmulBackend<T>::run(l_contiguous.get_data(), r_contiguous.get_data(), result.get_data(),
                              shape_left[0], shape_right[1], shape_left[1]);

        return result;
    }

    bool left_reshaped = false;
    bool right_reshaped = false;

    std::vector<std::size_t> shape_left(2);
    std::vector<std::size_t> strides_left(2);

    if (left.ndim() == 1) {
        left_reshaped = true;
        shape_left[0] = 1;
        shape_left[1] = left.get_shape()[0];
        strides_left[0] = 0;
        strides_left[1] = 1;
    } else {
        shape_left = left.get_shape();
    }

    std::vector<std::size_t> shape_right(2);
    std::vector<std::size_t> strides_right(2);

    if (right.ndim() == 1) {
        right_reshaped = true;
        shape_right[0] = right.get_shape()[0];
        shape_right[1] = 1;
        strides_right[0] = 1;
        strides_right[1] = 0;
    } else {
        shape_right = right.get_shape();
    }

    if (shape_left[shape_left.size() - 1] != shape_right[shape_right.size() - 2]) {
        throw std::invalid_argument("Incompatible matrix shapes.");
    }

    Tensor<T> l_contiguous = (left.is_contiguous()) ? left : as_contiguous_tensor(left);
    Tensor<T> r_contiguous = (right.is_contiguous()) ? right : as_contiguous_tensor(right);

    if (l_contiguous.ndim() != 1)
        strides_left = l_contiguous.get_strides();
    if (r_contiguous.ndim() != 1)
        strides_right = r_contiguous.get_strides();

    std::size_t batch_n_dim =
        (shape_right.size() > shape_left.size()) ? shape_right.size() - 2 : shape_left.size() - 2;

    std::vector<std::size_t> left_batch_shape(shape_left.begin(), shape_left.end() - 2);
    std::vector<std::size_t> right_batch_shape(shape_right.begin(), shape_right.end() - 2);

    if (!are_shapes_broadcastable(left_batch_shape, right_batch_shape)) {
        throw std::invalid_argument("Incompatible batch shapes");
    }

    std::vector<std::size_t> left_batch_strides(strides_left.begin(), strides_left.end() - 2);
    std::vector<std::size_t> right_batch_strides(strides_right.begin(), strides_right.end() - 2);
    std::vector<std::size_t> result_batch_shape;

    std::tie(result_batch_shape, left_batch_strides, right_batch_strides) =
        broadcast_shapes_and_strides(left_batch_shape, left_batch_strides, right_batch_shape,
                                     right_batch_strides);

    std::vector<std::size_t> result_shape = result_batch_shape;
    result_shape.push_back(shape_left[shape_left.size() - 2]);
    result_shape.push_back(shape_right[shape_right.size() - 1]);

    Tensor<T> result(result_shape);

    TensorIterator<T> iterator_left(l_contiguous.get_data(), result_batch_shape,
                                    left_batch_strides);

    TensorIterator<T> iterator_right(r_contiguous.get_data(), result_batch_shape,
                                     right_batch_strides);

    std::vector<std::size_t> result_batch_strides(result.get_strides().begin(),
                                                  result.get_strides().end() - 2);
    TensorIterator<T> iterator_result(result.get_data(), result_batch_shape, result_batch_strides);

    const std::size_t M = shape_left[shape_left.size() - 2];
    const std::size_t N = shape_right[shape_right.size() - 1];
    const std::size_t K = shape_left[shape_left.size() - 1];
    while (auto* out = iterator_result.next()) {
        auto* l = iterator_left.next();
        auto* r = iterator_right.next();

        MatmulBackend<T>::run(l, r, out, M, N, K);
    }

    if (right_reshaped || left_reshaped) {
        if (right_reshaped) {
            result.reshape_in_place(
                std::vector<std::size_t>(result_shape.begin(), result_shape.end() - 1));
        } else {
            result_shape.erase(result_shape.begin() + result_shape.size() - 2);
            result.reshape_in_place(result_shape);
        }
    }

    return result;
}

} // namespace pml
