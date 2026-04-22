#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <span>
#include <variant>
#include <algorithm>
#include <numeric>
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
    Tensor(const Tensor& other)
        : storage_(other.storage_), offset_(other.offset_), n_dim_(other.n_dim_),
          shape_(other.shape_), strides_(other.strides_), data_num_elems_(other.data_num_elems_),
          is_view_(other.is_view_) {}
    Tensor& operator=(const Tensor&) = default;

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

    Tensor<T> view(const std::vector<std::size_t>& shape) const {
        if (!is_contiguous()) {
            throw std::logic_error("Memory is not contiguous to make a view.");
        }

        std::vector<std::size_t> new_strides;
        new_strides.resize(shape.size());
        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= shape[i];
        }

        if (stride != data_num_elems_) {
            throw std::invalid_argument("Incorrect view shape.");
        }

        return Tensor<T>(storage_, offset_, shape, new_strides);
    }

    void reshape_in_place(const std::vector<std::size_t>& shape) {
        if (!is_contiguous()) {
            throw std::logic_error("Memory is not contiguous to reshape tensor.");
        }
        if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>()) !=
            data_num_elems_) {
            throw std::invalid_argument("Incorrect shape.");
        }

        shape_ = shape;
        n_dim_ = shape.size();
        calculate_strides();
    }

    // Overload operator<< for printing
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }

    template <typename... Args> Tensor<T> operator[](Args&&... args) const {
        using Index = std::variant<std::ptrdiff_t, Slice>;

        if (sizeof...(Args) > n_dim_) {
            throw std::invalid_argument("Too many arguments.");
        }

        std::vector<Index> indices;
        indices.reserve(sizeof...(Args));
        (indices.emplace_back(std::forward<Args>(args)), ...);

        std::size_t new_offset = offset_;
        std::vector<std::size_t> new_shape{};
        std::vector<std::size_t> new_strides{};

        std::size_t arg_idx = 0;
        for (std::size_t dim = 0; dim < n_dim_; dim++) {
            if (arg_idx < indices.size() &&
                std::holds_alternative<std::ptrdiff_t>(indices[arg_idx])) {
                std::ptrdiff_t idx = std::get<std::ptrdiff_t>(indices[arg_idx]);
                if (idx < 0)
                    idx += static_cast<std::ptrdiff_t>(shape_[dim]);

                if (idx < 0 || static_cast<std::size_t>(idx) >= shape_[dim]) {
                    throw std::out_of_range("Index is out of range.");
                }

                new_offset += static_cast<std::size_t>(idx) * strides_[dim];
                arg_idx++;
            } else {
                Slice slc(0, -1, 1);
                if (arg_idx < indices.size()) {
                    slc = std::get<Slice>(indices[arg_idx]);
                    arg_idx++;
                }

                std::ptrdiff_t start = (slc.start < 0)
                                           ? slc.start + static_cast<std::ptrdiff_t>(shape_[dim])
                                           : slc.start;
                std::ptrdiff_t end = (slc.end < 0)
                                         ? slc.end + static_cast<std::ptrdiff_t>(shape_[dim]) + 1
                                         : slc.end;

                if (start < 0 || start > end || static_cast<std::size_t>(end) > shape_[dim] ||
                    slc.step == 0) {
                    throw std::out_of_range("Invalid slice.");
                }

                std::size_t slice_len =
                    static_cast<std::size_t>((end - start + slc.step - 1) / slc.step);

                new_shape.push_back(slice_len);
                new_strides.push_back(strides_[dim] * slc.step);
                new_offset += static_cast<std::size_t>(start) * strides_[dim];
            }
        }

        return Tensor<T>(storage_, new_offset, new_shape, new_strides);
    }

    const std::vector<std::size_t>& get_shape() const noexcept {
        return shape_;
    }

    const std::vector<std::size_t>& get_strides() const noexcept {
        return strides_;
    }

    const std::size_t ndim() const noexcept {
        return n_dim_;
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

} // namespace pml
