#include <iostream>
#include <pml/core/tensor.hpp>
#include <pml/core/operations.hpp>

int main() {
    pml::Tensor<float> t({{1, 2, 3, 4}}, {2, 2});
    std::cout << t + pml::Tensor(1.f) << std::endl;
    return 0;
}
