#include <tensor.hpp>
#include <iostream>

int main() {
    float data[] = {
        -0.24670549,
        1.64457978,
        -1.9741151,
    };
    pml::Tensor<float> t1(data, sizeof(data) / sizeof(float), {3});
    std::cout << t1 << std::endl;

    pml::Tensor<float> t3 = pml::sum(t1, 0, false);
    std::cout << t3 << std::endl;
    return 0;
}
