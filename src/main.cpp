#include <matmul.hpp>
#include <slice.hpp>
#include <tensor.hpp>
#include <operations.hpp>
#include <iostream>

int main() {
    pml::Tensor<float> t1({5000, 5000});

    float data2[] = {};
    pml::Tensor<float> t2({5000, 5000});

    pml::Tensor<float> t_res = pml::add(t2, t2);
    return 0;
}
