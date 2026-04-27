#include <matmul.hpp>
#include <slice.hpp>
#include <tensor.hpp>
#include <operations.hpp>
#include <iostream>
#include <math_kernels.h>

int main() {
    alignas(32) float data[] = {1., 2., 3., 4., 5., 6., 7., 8., 8.,
                                7., 6., 5., 4., 3., 2., 1., 2., 1.};
    exp_avx2_aligned(data, data, 18);

    for (std::size_t i = 0; i < 18; i++)
        std::cout << data[i] << ' ';
    std::cout << std::endl;
    return 0;
}
