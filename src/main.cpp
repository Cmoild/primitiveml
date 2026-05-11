#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>
#include <math_kernels.h>
#include <matmul.hpp>

int main() {
    pml::Tensor<int> t({{1, 2, 3, 4}}, {2, 2});
    std::cout << t << std::endl;
    std::cout << t.transpose(0, 1) << std::endl;
    return 0;
}
