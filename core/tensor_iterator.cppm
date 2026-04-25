export module pml:iterator;

import std;
import :tensor;

export namespace pml {

template <typename T> class TensorIterator {
  private:
    std::vector<std::size_t> current_indices_;
    T* data_ptr_;
    std::vector<std::size_t> strides_;
    std::vector<std::size_t> shape_;
    std::size_t global_index_;
    std::size_t num_elements_;
    std::size_t contiguous_block_size_;
    bool finished_;
    bool started_;

  public:
    // Constructor
    TensorIterator(T* data, const std::vector<std::size_t>& shape,
                   const std::vector<std::size_t>& strides)
        : data_ptr_(data), shape_(shape), strides_(strides), finished_(false), started_(false),
          global_index_(0) {
        current_indices_.resize(shape.size(), 0);
        num_elements_ =
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());

        std::size_t expected_stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            if (strides[i] != expected_stride)
                break;

            expected_stride *= shape[i];
        }
        contiguous_block_size_ = expected_stride;
    }

    // Get next element
    T* next() {
        if (finished_) {
            return nullptr;
        }
        if (!started_) {
            started_ = true;
            global_index_++;
            return data_ptr_;
        }
        if (global_index_ >= num_elements_) {
            finished_ = true;
            return nullptr;
        }

        // Increment indices
        for (std::ptrdiff_t i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < shape_[i]) {
                break;
            }
            current_indices_[i] = 0;
            if (i == 0) {
                finished_ = true;
                return nullptr;
            }
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            offset += current_indices_[i] * strides_[i];
        }

        global_index_++;

        return data_ptr_ + offset;
    }

    T* next_contiguous_block() {
        if (finished_) {
            return nullptr;
        }
        if (!started_) {
            started_ = true;
            global_index_ += contiguous_block_size_;
            return data_ptr_;
        }
        if (global_index_ >= num_elements_) {
            finished_ = true;
            return nullptr;
        }

        // Increment indices
        std::size_t carry = contiguous_block_size_;
        for (std::ptrdiff_t i = current_indices_.size() - 1; i >= 0; --i) {
            std::size_t sum = current_indices_[i] + carry;

            current_indices_[i] = sum % shape_[i];
            carry = sum / shape_[i];

            if (carry == 0) {
                break;
            }

            if (i == 0) {
                finished_ = true;
                return nullptr;
            }
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            offset += current_indices_[i] * strides_[i];
        }

        global_index_ += contiguous_block_size_;

        return data_ptr_ + offset;
    }

    T* advance(std::size_t n) {
        if (contiguous_block_size_ % n != 0) {
            throw std::invalid_argument("Incorrect iterator step.");
        }
        if (finished_) {
            return nullptr;
        }
        if (!started_) {
            started_ = true;
            global_index_ += n;
            return data_ptr_;
        }
        if (global_index_ >= num_elements_) {
            finished_ = true;
            return nullptr;
        }

        // Increment indices
        std::size_t carry = n;
        for (std::ptrdiff_t i = current_indices_.size() - 1; i >= 0; --i) {
            std::size_t sum = current_indices_[i] + carry;

            current_indices_[i] = sum % shape_[i];
            carry = sum / shape_[i];

            if (carry == 0) {
                break;
            }

            if (i == 0) {
                finished_ = true;
                return nullptr;
            }
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            offset += current_indices_[i] * strides_[i];
        }

        global_index_ += n;

        return data_ptr_ + offset;
    }

    // Get element by index
    T* at(std::size_t idx) const {
        std::vector<std::size_t> indices(shape_.size(), 0);
        for (std::ptrdiff_t i = shape_.size() - 1; i >= 0; --i) {
            indices[i] = idx % shape_[i];
            idx /= shape_[i];
        }
        if (idx > 0) {
            throw std::out_of_range("Index out of bounds.");
        }

        // Calculate offset
        std::size_t offset = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return data_ptr_ + offset;
    }

    // Check if iteration is finished
    bool is_finished() const noexcept {
        return finished_;
    }

    std::size_t contiguous_block_size() const noexcept {
        return contiguous_block_size_;
    }

    // Reset iterator
    void reset() {
        std::fill(current_indices_.begin(), current_indices_.end(), 0);
        finished_ = false;
        started_ = false;
    }
};

} // namespace pml
