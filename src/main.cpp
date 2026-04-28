#include <cstddef>
#include <matmul.hpp>
#include <slice.hpp>
#include <tensor.hpp>
#include <operations.hpp>
#include <iostream>
#include <math_kernels.h>

const std::size_t data_size = 5000000;
static std::clock_t total_aligned = 0;
static std::clock_t total_unaligned = 0;

void init_data(float* data) {
    for (std::size_t i = 0; i < data_size; i++)
        data[i] = static_cast<float>(rand() % 1000 - 500) / 100;
}

void run_aligned(float* data, float* out) {
    std::clock_t start = clock();
    for (std::size_t i = 0; i < 1000; i++) {
        exp_avx2_aligned(data, out, data_size);
    }
    std::clock_t end = clock();

    std::cout << "Aligned: " << end - start << std::endl;

    total_aligned += end - start;
}

void run_unaligned(float* data, float* out) {
    std::clock_t start = clock();
    for (std::size_t i = 0; i < 1000; i++) {
        exp_avx2_unaligned(data, out, data_size);
    }
    std::clock_t end = clock();

    std::cout << "Unaligned: " << end - start << std::endl;

    total_unaligned += end - start;
}

int main() {
    float* data = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * data_size));
    float* out = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * data_size));

    init_data(data);

    for (std::size_t i = 0; i < 10; i++) {
        if (rand() % 2 == 0) {
            run_aligned(data, out);
            run_unaligned(data, out);
        } else {
            run_unaligned(data, out);
            run_aligned(data, out);
        }

        if (i < 2) {
            total_aligned = 0;
            total_unaligned = 0;
        }
    }

    std::cout << "Avg aligned: " << total_aligned / 8 << std::endl;
    std::cout << "Avg unaligned: " << total_unaligned / 8 << std::endl;

    free(data);
    free(out);
    return 0;
}
