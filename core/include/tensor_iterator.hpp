#pragma once

#include <tensor.hpp>
#include <numeric>

namespace pml {

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

} // namespace pml
