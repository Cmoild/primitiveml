#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>

int main() {
    int inp[] = {1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4, 0};
    pml::Tensor<int> t(inp, {3, 4});
    std::cout << t[pml::Slice(0, -1, 2)] + pml::Tensor(1) << std::endl;
    return 0;
}
