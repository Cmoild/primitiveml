#include <matmul.hpp>
#include <slice.hpp>
#include <tensor.hpp>
#include <iostream>

int main() {
    float data1[] = {-0.7499, 0.3517, 1.0287, 0.3540, 0.6332, 0.5136, 3.0881, 0.7850, 0.0154};
    pml::Tensor<float> t1(data1, {3, 3});
    std::cout << t1 << std::endl;

    float data2[] = {0.9418, -0.0170, -0.2001, -1.9168, 1.2530, 0.3503, 0.7031, -0.8394, -1.0559};
    pml::Tensor<float> t2(data2, {3, 3});
    t2 = t2[1];
    std::cout << t2 << std::endl;

    pml::Tensor<float> t_res = pml::matmul(t2, t1);
    std::cout << t_res << std::endl;
    return 0;
}
