#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>
#include <math_kernels.h>
#include <matmul.hpp>

int main() {
    pml::Tensor<float> t({{-1.f, 2.f, 3.f, -4.f, 5.f, 6.f}}, {6});
    std::cout << pml::relu(t) << std::endl;
    return 0;
}
