#include <cstddef>
#include <iostream>
#include <math_kernels.h>
#include <ctime>

const std::size_t data_size = 500000;
static std::clock_t total_intrin = 0;
static std::clock_t total_unaligned = 0;

void init_data(float* data) {
    for (std::size_t i = 0; i < data_size; i++)
        data[i] = static_cast<float>(rand() % 1000 - 500) / 100;
}

void run_unaligned(float* data, float* out) {
    std::clock_t start = clock();
    float checksum = 0;
    for (std::size_t i = 0; i < 1000; i++) {
        exp_avx2(data, out, data_size);
        checksum += out[i];
    }
    std::clock_t end = clock();

    std::cout << "Unaligned: " << end - start << "\tchecksum: " << checksum << std::endl;

    total_unaligned += end - start;
}

void run_intrin(float* data, float* out) {
    std::clock_t start = clock();
    float checksum = 0;
    for (std::size_t i = 0; i < 1000; i++) {
        exp_avx2_intrin(data, out, data_size);
        checksum += out[i];
    }
    std::clock_t end = clock();

    std::cout << "Intrin: " << end - start << "\t\tchecksum: " << checksum << std::endl;

    total_intrin += end - start;
}

int main() {
    float* data = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * data_size));
    float* out = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * data_size));

    init_data(data);

    for (std::size_t i = 0; i < 30; i++) {

        if (rand() % 2 == 0) {
            run_unaligned(data, out);
            run_intrin(data, out);
        } else {
            run_intrin(data, out);
            run_unaligned(data, out);
        }
        if (i < 2) {
            total_intrin = 0;
            total_unaligned = 0;
        }
    }

    std::cout << "Avg intrin: " << total_intrin / 28 << std::endl;
    std::cout << "Avg unaligned: " << total_unaligned / 28 << std::endl;

    free(data);
    free(out);
    return 0;
}
