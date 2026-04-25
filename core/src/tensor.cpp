module pml;

import std;
import :tensor;

namespace pml {

bool are_shapes_broadcastable(const std::vector<std::size_t>& left_shape,
                              const std::vector<std::size_t>& right_shape) {
    for (std::size_t i = 0; i < left_shape.size() && i < right_shape.size(); ++i) {
        const std::size_t left_dim = left_shape[left_shape.size() - 1 - i];
        const std::size_t right_dim = right_shape[right_shape.size() - 1 - i];

        if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
            return false;
        }
    }
    return true;
}

std::tuple<std::vector<std::size_t>, std::vector<std::size_t>, std::vector<std::size_t>>
broadcast_shapes_and_strides(const std::vector<std::size_t>& left_shape,
                             const std::vector<std::size_t>& left_strides,
                             const std::vector<std::size_t>& right_shape,
                             const std::vector<std::size_t>& right_strides) {
    if (left_shape.size() != left_strides.size() || right_shape.size() != right_strides.size()) {
        throw std::invalid_argument("Shape and strides sizes must match.");
    }

    const std::size_t max_n_dim = std::max(left_shape.size(), right_shape.size());

    std::vector<std::size_t> result_shape(max_n_dim, 1);
    std::vector<std::size_t> result_left_strides(max_n_dim, 0);
    std::vector<std::size_t> result_right_strides(max_n_dim, 0);

    for (std::size_t i = 0; i < max_n_dim; ++i) {
        const std::size_t out_idx = max_n_dim - 1 - i;

        const bool has_left = i < left_shape.size();
        const bool has_right = i < right_shape.size();

        const std::size_t left_dim = has_left ? left_shape[left_shape.size() - 1 - i] : 1;
        const std::size_t right_dim = has_right ? right_shape[right_shape.size() - 1 - i] : 1;

        if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }

        result_shape[out_idx] = std::max(left_dim, right_dim);

        if (has_left) {
            const std::size_t left_stride = left_strides[left_strides.size() - 1 - i];
            result_left_strides[out_idx] = (left_dim == 1 ? 0 : left_stride);
        }

        if (has_right) {
            const std::size_t right_stride = right_strides[right_strides.size() - 1 - i];
            result_right_strides[out_idx] = (right_dim == 1 ? 0 : right_stride);
        }
    }

    return {result_shape, result_left_strides, result_right_strides};
}

} // namespace pml
