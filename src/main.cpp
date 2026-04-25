import pml;
import std;

int main() {
    pml::Tensor<float> t1({500, 500});

    pml::Tensor<float> t2({500, 500});

    pml::Tensor<float> t_res = pml::add(t2, t2);
    return 0;
}
