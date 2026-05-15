#pragma once

#include <cstddef>
#include <cstdlib>
#include <source_location>
#include <exceptions.hpp>

namespace pml {

class Allocator {
  public:
    virtual void* allocate(const std::size_t nbytes, const std::size_t alignment) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual ~Allocator() = default;
};

class MallocAllocator : public Allocator {
    void* allocate(const std::size_t nbytes, const std::size_t alignment) override {
        void* out = std::malloc(nbytes);
        if (!out)
            throw custom_bad_alloc("Failed to allocate memory.", std::source_location::current());
        return out;
    }

    void deallocate(void* ptr) override {
        std::free(ptr);
    }
};

class Storage {
  private:
    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    Allocator* allocator_ = nullptr;

  public:
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_), allocator_(other.allocator_) {
        other.data_ = nullptr;
        other.nbytes_ = 0;
    }
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (data_ && allocator_)
                allocator_->deallocate(data_);
            data_ = other.data_;
            nbytes_ = other.nbytes_;
            allocator_ = other.allocator_;
            other.data_ = nullptr;
            other.nbytes_ = 0;
        }
        return *this;
    }

    Storage(const std::size_t nbytes, Allocator& allocator, const std::size_t alignment = 0)
        : nbytes_(nbytes), allocator_(&allocator) {
        data_ = allocator_->allocate(nbytes, alignment);
    }

    ~Storage() {
        if (data_ && allocator_)
            allocator_->deallocate(data_);
        // TODO: Add `throw runtime_error`
    };

    void* data() const noexcept {
        return data_;
    }

    std::size_t nbytes() const noexcept {
        return nbytes_;
    }
};

} // namespace pml
