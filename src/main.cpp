#include <slice.hpp>
#include <tensor.hpp>
#include <iostream>

int main() {
    float data[] = {-0.48054148, -1.41560552, -0.81768925, -0.49359604, -1.13949375, 0.56858589,
                    0.04630868,  0.5937672,   -1.80612649, 0.79860799,  0.04094044,  -0.54713281,
                    -1.80040832, 0.03186612,  1.154988,    -0.14142765, 0.70132991,  1.08274669,
                    -0.91094437, 0.21639038,  -1.29521017, -1.54521664, 0.91282814,  -0.24690078};
    pml::Tensor<float> t(data, {2, 3, 4});
    std::cout << t << std::endl;

    pml::Tensor<float> t_view = t[pml::Slice(0, 2), pml::Slice(0, 2), pml::Slice(0, -1, 2)];
    std::cout << t_view << std::endl;
    return 0;
}
