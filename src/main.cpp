#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>
#include <math_kernels.h>
#include <matmul.hpp>

int main() {
    pml::Tensor<int> t({{1, 2, 3, 4}}, {2, 2});
    pml::Tensor<int> t_dest({{0, 0, 0, 0}}, {4});
    pml::inplace_copy(t_dest[pml::Slice(0, 2)], t[1]);
    std::cout << t << std::endl;
    std::cout << t_dest << std::endl;
    pml::inplace_copy(t_dest[pml::Slice(1, 4, 2)], t.transpose(0, 1)[0]);
    std::cout << t_dest << std::endl;
    return 0;
}
