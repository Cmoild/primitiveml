#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace pml {

template <typename T> class Tensor {
  private:
    std::size_t n_dim;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    std::unique_ptr<T[]> data;
    std::size_t data_num_elems;
    bool is_view;

  public:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Constructor with data copy
    Tensor(const T* input_data, std::size_t data_len, const std::vector<std::size_t>& shape)
        : n_dim(shape.size()), shape(shape), is_view(false) {
        if (data_len == 0 || shape.empty()) {
            throw std::invalid_argument("Invalid tensor dimensions or data length.");
        }

        // Calculate strides and allocate memory
        calculate_strides();
        data_num_elems = calculate_num_elements();
        data = std::make_unique<T[]>(data_num_elems);
        std::copy(input_data, input_data + data_num_elems, data.get());
    }

    // Constructor with empty data
    Tensor(const std::vector<std::size_t>& shape)
        : n_dim(shape.size()), shape(shape), is_view(false) {
        if (shape.empty()) {
            throw std::invalid_argument("Invalid tensor dimensions.");
        }

        // Calculate strides and allocate memory
        calculate_strides();
        data_num_elems = calculate_num_elements();
        data = std::make_unique<T[]>(data_num_elems);
    }

    // Constructor without data copy
    Tensor(T* input_data, std::size_t data_len, const std::vector<std::size_t>& shape, bool is_view)
        : n_dim(shape.size()), shape(shape), is_view(is_view) {
        if (data_len == 0 || shape.empty()) {
            throw std::invalid_argument("Invalid tensor dimensions or data length.");
        }

        // Calculate strides
        calculate_strides();
        data_num_elems = calculate_num_elements();
        if (is_view) {
            data.reset(input_data, [](T*) {}); // No ownership
        } else {
            data = std::make_unique<T[]>(data_num_elems);
            std::copy(input_data, input_data + data_num_elems, data.get());
        }
    }

    // Destructor
    ~Tensor() = default;

    // Print method
    void print(std::ostream& os = std::cout) const {
        os << "Tensor: {\n";
        print_recursive(os, 0, data.get());
        os << "\nShape: [";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            os << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        os << "]\nStrides: [";
        for (std::size_t i = 0; i < strides.size(); ++i) {
            os << strides[i] << (i < strides.size() - 1 ? ", " : "");
        }
        os << "]\n}";
    }

    // Overload operator<< for printing
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }

    const std::vector<std::size_t>& get_shape() const noexcept {
        return shape;
    }

    const std::vector<std::size_t>& get_strides() const noexcept {
        return strides;
    }

    T* get_data() const noexcept {
        return data.get();
    }

    std::size_t get_data_num_elems() const noexcept {
        return data_num_elems;
    }

    bool is_contiguous() const noexcept {
        std::size_t expected = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            if (strides[i] != expected)
                return false;
            expected *= shape[i];
        }
        return true;
    }

  private:
    void calculate_strides() {
        strides.resize(n_dim);
        std::size_t stride = 1;
        for (std::size_t i = n_dim; i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    std::size_t calculate_num_elements() const {
        std::size_t num_elems = 1;
        for (std::size_t dim : shape) {
            num_elems *= dim;
        }
        return num_elems;
    }

    void print_recursive(std::ostream& os, std::size_t shape_idx, const T* data_ptr) const {
        if (shape_idx == shape.size() - 1) {
            os << "[ ";
            for (std::size_t i = 0; i < shape[shape_idx]; ++i) {
                os << data_ptr[i * strides[shape_idx]] << (i < shape[shape_idx] - 1 ? " " : "");
            }
            os << " ]";
        } else {
            os << "[";
            for (std::size_t i = 0; i < shape[shape_idx]; ++i) {
                print_recursive(os, shape_idx + 1, data_ptr + i * strides[shape_idx]);
                if (i < shape[shape_idx] - 1) {
                    os << ",\n";
                }
            }
            os << "]";
        }
    }
};

bool are_shapes_broadcastable(const std::vector<std::size_t>& left_shape,
                              const std::vector<std::size_t>& right_shape);

std::tuple<std::vector<std::size_t>, std::vector<std::size_t>, std::vector<std::size_t>>
broadcast_shapes_and_strides(const std::vector<std::size_t>& left_shape,
                             const std::vector<std::size_t>& left_strides,
                             const std::vector<std::size_t>& right_shape,
                             const std::vector<std::size_t>& right_strides);

template <typename T> class TensorIterator {
  private:
    std::vector<std::size_t> current_indices;
    T* data_ptr;
    std::vector<std::size_t> strides;
    std::vector<std::size_t> shape;
    bool finished;
    bool started;

  public:
    // Constructor
    TensorIterator(T* data, const std::vector<std::size_t>& shape,
                   const std::vector<std::size_t>& strides)
        : data_ptr(data), shape(shape), strides(strides), finished(false), started(false) {
        if (shape.empty() || strides.empty()) {
            throw std::invalid_argument("Shape and strides must not be empty.");
        }
        current_indices.resize(shape.size(), 0);
    }

    // Get next element
    T* next() {
        if (finished) {
            return nullptr;
        }
        if (!started) {
            started = true;
            return data_ptr;
        }

        // Increment indices
        for (std::ptrdiff_t i = current_indices.size() - 1; i >= 0; --i) {
            if (++current_indices[i] < shape[i]) {
                break;
            }
            current_indices[i] = 0;
            if (i == 0) {
                finished = true;
                return nullptr;
            }
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < current_indices.size(); ++i) {
            offset += current_indices[i] * strides[i];
        }
        return data_ptr + offset;
    }

    // Get element by index
    T* at(std::size_t idx) const {
        std::vector<std::size_t> indices(shape.size(), 0);
        for (std::ptrdiff_t i = shape.size() - 1; i >= 0; --i) {
            indices[i] = idx % shape[i];
            idx /= shape[i];
        }
        if (idx > 0) {
            throw std::out_of_range("Index out of bounds.");
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides[i];
        }
        return data_ptr + offset;
    }

    // Check if iteration is finished
    bool is_finished() const noexcept {
        return finished;
    }

    // Reset iterator
    void reset() {
        std::fill(current_indices.begin(), current_indices.end(), 0);
        finished = false;
        started = false;
    }
};

// Elementwise operation template
template <typename T, typename Op>
inline Tensor<T> elementwise_operation(const Tensor<T>& left, const Tensor<T>& right, Op op) {
    if (!are_shapes_broadcastable(left.get_shape(), right.get_shape())) {
        throw std::invalid_argument("Tensors are not broadcastable.");
    }

    auto [result_shape, left_strides, right_strides] = broadcast_shapes_and_strides(
        left.get_shape(), left.get_strides(), right.get_shape(), right.get_strides());

    Tensor<T> result(result_shape);

    TensorIterator<T> left_iter(left.get_data(), result_shape, left_strides);
    TensorIterator<T> right_iter(right.get_data(), result_shape, right_strides);
    TensorIterator<T> result_iter(result.get_data(), result.get_shape(), result.get_strides());

    for (;;) {
        auto* l = left_iter.next();
        auto* r = right_iter.next();
        auto* out = result_iter.next();

        if (result_iter.is_finished())
            break;

        *out = op(*l, *r);
    };

    return result;
}

template <typename T> inline T add_operation(const T& left, const T& right) {
    return left + right;
}

template <typename T> Tensor<T> add(const Tensor<T>& left, const Tensor<T>& right) {
    return elementwise_operation(left, right, add_operation<T>);
}

} // namespace pml
