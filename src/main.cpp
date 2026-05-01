#include <cpu_flags.h>
#include <iostream>

int main() {
    std::cout << "Must be 1: " << avx2_fma_supported() << std::endl;
    return 0;
}
