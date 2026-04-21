#pragma once

#include <cstddef>
#include <stdexcept>
#include <tensor.hpp>

namespace pml {

template <typename T> Tensor<T> matmul(const Tensor<T>& left, const Tensor<T>& right) {
    if (left.ndim() == 0 || right.ndim() == 0) {
        throw std::invalid_argument("Tensor shapes are incompatible with matmul op.");
    }

    if (left.ndim() <= 2 && right.ndim() <= 2) {
        bool left_reshaped = false;
        bool right_reshaped = false;

        std::vector<std::size_t> shape_left(2);
        std::vector<std::size_t> strides_left(2);

        if (left.ndim() == 1) {
            left_reshaped = true;
            shape_left[0] = 1;
            shape_left[1] = left.get_shape()[0];
            strides_left[0] = 0;
            strides_left[1] = left.get_strides()[0];
        } else {
            shape_left = left.get_shape();
            strides_left = left.get_strides();
        }

        std::vector<std::size_t> shape_right(2);
        std::vector<std::size_t> strides_right(2);

        if (right.ndim() == 1) {
            right_reshaped = true;
            shape_right[0] = right.get_shape()[0];
            shape_right[1] = 1;
            strides_right[0] = right.get_strides()[0];
            strides_right[1] = 0;
        } else {
            shape_right = right.get_shape();
            strides_right = right.get_strides();
        }

        if (shape_left[1] != shape_right[0]) {
            throw std::invalid_argument("Incompatible matrix shapes.");
        }
    }
}

} // namespace pml
