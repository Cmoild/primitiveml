#pragma once

#include <functional>
#include <numeric>
#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <span>
#include <variant>
#include <slice.hpp>
#include <storage.hpp>

namespace pml {

template <typename T> class Tensor {
  private:
    std::shared_ptr<Storage> storage_;
    std::size_t offset_ = 0;
    std::size_t n_dim_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t data_num_elems_;
    bool is_view_ = false;

  public:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Constructor with data copy
    Tensor(std::span<const T> input_data, const std::vector<std::size_t>& shape)
        : n_dim_(shape.size()), shape_(shape), is_view_(false) {
        calculate_strides();
        data_num_elems_ = calculate_num_elements();

        storage_ = make_storage(data_num_elems_ * sizeof(T));
        std::copy(input_data.begin(), input_data.end(), get_data());
    }

    // Constructor with scalar value
    Tensor(const T scalar_value)
        : n_dim_(0), shape_({}), is_view_(false), data_num_elems_(1), strides_({}) {
        storage_ = make_storage(data_num_elems_ * sizeof(T));
        *get_data() = scalar_value;
    }

    // Constructor with empty data
    Tensor(const std::vector<std::size_t>& shape)
        : n_dim_(shape.size()), shape_(shape), is_view_(false) {
        calculate_strides();
        data_num_elems_ = calculate_num_elements();
        storage_ = make_storage(data_num_elems_ * sizeof(T));
    }

    // Constructor without data copy (moves ownership to tensor)
    Tensor(std::unique_ptr<Storage> input_data, const std::vector<std::size_t>& shape)
        : n_dim_(shape.size()), shape_(shape), is_view_(false) {
        calculate_strides();
        data_num_elems_ = calculate_num_elements();
        storage_ = std::move(input_data);
    }

    // Constructor without data copy (view)
    Tensor(std::shared_ptr<Storage> input_data, std::size_t offset,
           const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides)
        : n_dim_(shape.size()), shape_(shape), is_view_(true), strides_(strides), offset_(offset) {
        data_num_elems_ = calculate_num_elements();
        storage_ = input_data;
    }

    // Destructor
    ~Tensor() = default;

    // Print method
    void print(std::ostream& os = std::cout) const {
        if (shape_.size() == 0) {
            os << "Scalar: " << *get_data() << "\n";
            return;
        }
        os << "Tensor: {\n";
        print_recursive(os, 0, get_data());
        os << "\nShape: [";
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            os << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
        }
        os << "]\nStrides: [";
        for (std::size_t i = 0; i < strides_.size(); ++i) {
            os << strides_[i] << (i < strides_.size() - 1 ? ", " : "");
        }
        os << "]\n}";
    }

    // Overload operator<< for printing
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }

    template <typename... Args> Tensor<T> operator[](Args... args) {
        std::variant<std::ptrdiff_t, Slice> indices(args...);
        std::vector<std::size_t> slice_shape{};
        std::vector<std::size_t> slice_strides{};
    }

    const std::vector<std::size_t>& get_shape() const noexcept {
        return shape_;
    }

    const std::vector<std::size_t>& get_strides() const noexcept {
        return strides_;
    }

    T* get_data() const noexcept {
        return static_cast<T*>(storage_->data()) + offset_;
    }

    std::size_t get_data_num_elems() const noexcept {
        return data_num_elems_;
    }

    bool is_contiguous() const noexcept {
        std::size_t expected = 1;
        for (std::size_t i = shape_.size(); i-- > 0;) {
            if (strides_[i] != expected)
                return false;
            expected *= shape_[i];
        }
        return true;
    }

  private:
    void calculate_strides() {
        strides_.resize(n_dim_);
        std::size_t stride = 1;
        for (std::size_t i = n_dim_; i-- > 0;) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    std::size_t calculate_num_elements() const {
        std::size_t num_elems = 1;
        for (std::size_t dim : shape_) {
            num_elems *= dim;
        }
        return num_elems;
    }

    void print_recursive(std::ostream& os, std::size_t shape_idx, const T* data_ptr) const {
        if (shape_idx == shape_.size() - 1) {
            os << "[ ";
            for (std::size_t i = 0; i < shape_[shape_idx]; ++i) {
                os << data_ptr[i * strides_[shape_idx]] << (i < shape_[shape_idx] - 1 ? " " : "");
            }
            os << " ]";
        } else {
            os << "[";
            for (std::size_t i = 0; i < shape_[shape_idx]; ++i) {
                print_recursive(os, shape_idx + 1, data_ptr + i * strides_[shape_idx]);
                if (i < shape_[shape_idx] - 1) {
                    os << ",\n";
                }
            }
            os << "]";
        }
    }

    std::shared_ptr<Storage> make_storage(std::size_t nbytes, std::size_t alignment = 0) {
        static MallocAllocator default_allocator;
        return std::make_shared<Storage>(nbytes, default_allocator, alignment);
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
    std::size_t global_index;
    std::size_t num_elements;
    bool finished;
    bool started;

  public:
    // Constructor
    TensorIterator(T* data, const std::vector<std::size_t>& shape,
                   const std::vector<std::size_t>& strides)
        : data_ptr(data), shape(shape), strides(strides), finished(false), started(false),
          global_index(0) {
        current_indices.resize(shape.size(), 0);
        num_elements =
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    }

    // Get next element
    T* next() {
        if (finished) {
            return nullptr;
        }
        if (!started) {
            started = true;
            global_index++;
            return data_ptr;
        }
        if (global_index >= num_elements) {
            finished = true;
            return nullptr;
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

        global_index++;

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

    while (auto* out = result_iter.next()) {
        auto* l = left_iter.next();
        auto* r = right_iter.next();

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

} // namespace pml
