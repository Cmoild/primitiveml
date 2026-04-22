#include <matmul.hpp>
#include <slice.hpp>
#include <tensor.hpp>
#include <operations.hpp>
#include <iostream>

int main() {
    float data1[] = {-8.5520e-01, 3.9847e-01,  2.0572e+00,  8.1047e-01,  -7.4922e-01, -7.3988e-01,
                     -2.9663e-01, -2.0376e+00, -3.5933e-01, 4.9670e-06,  7.8602e-02,  -1.1403e+00,
                     -4.6712e-02, 7.2534e-01,  2.0451e+00,  -1.2321e+00, 3.7795e-01,  -6.4778e-03,
                     -7.3191e-02, 8.4399e-01,  3.1209e-01,  5.5157e-02,  7.0616e-01,  5.0995e-01};
    pml::Tensor<float> t1(data1, {2, 3, 4});
    std::cout << t1 << std::endl;

    float data2[] = {-0.2807, 1.1287, 1.7838, 1.6381};
    pml::Tensor<float> t2(data2, {4});
    std::cout << t2 << std::endl;

    pml::Tensor<float> t_res = pml::matmul(t2, t2);
    std::cout << t_res << std::endl;
    return 0;
}
