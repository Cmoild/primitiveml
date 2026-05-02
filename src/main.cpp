#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>

int main() {
    pml::Tensor<float> t({10000, 10000});
    pml::Tensor<float> t2({10000, 10000});
    auto t3 = pml::add(t, t2);
    std::cout << t3[0, pml::Slice(0, 10)];
    return 0;
}
