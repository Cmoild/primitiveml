#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>

int main() {
    float inp[] = {1.1, 2.1, 3.2, 4.3};
    pml::Tensor<float> t1(inp, {2, 2});
    std::cout << t1 << std::endl;
    pml::Tensor<int> t2 = pml::cast<int>(t1);
    std::cout << t2 << std::endl;
    return 0;
}
