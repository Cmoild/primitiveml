#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>

int main() {
    float inp[] = {1.1, 2.1, 3.2, 4.3};
    pml::Tensor<float> t1(inp, {2, 2});
    std::cout << t1 << std::endl;
    pml::Tensor<float> t2 = (t1 + pml::Tensor(2.f) * pml::Tensor(3.f)) / pml::Tensor(10.f);
    std::cout << t2 << std::endl;
    return 0;
}
