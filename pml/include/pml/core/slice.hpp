#pragma once

#include <cstddef>
namespace pml {

struct Slice {
    std::ptrdiff_t start = 0;
    std::ptrdiff_t end = -1;
    std::size_t step = 1;

    Slice() = default;
    Slice(std::ptrdiff_t start, std::ptrdiff_t end, std::size_t step)
        : start(start), end(end), step(step) {}
    Slice(std::ptrdiff_t start, std::ptrdiff_t end) : start(start), end(end), step(1) {}
};

} // namespace pml
