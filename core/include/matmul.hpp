#pragma once

#include <cstddef>
#include <stdexcept>
#include <tensor.hpp>

namespace pml {

template <typename T>
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

template <typename T> Tensor<T> matmul(const Tensor<T>& left, const Tensor<T>& right) {
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

        matmul_kernel(l_contiguous.get_data(), r_contiguous.get_data(), result.get_data(),
                      shape_left[0], shape_right[1], shape_left[1]);

        return result;
    }

    return Tensor<T>({6, 7, 67});
}

} // namespace pml
