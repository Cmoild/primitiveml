#include <tensor.hpp>
#include <iostream>

int main() {
    pml::Tensor<float> t1(10.f);
    pml::Tensor<float> t2(5.f);

    pml::Tensor<float> t3 = pml::add(t1, t2);
    std::cout << t3 << std::endl;
    return 0;
}
