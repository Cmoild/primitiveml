#include <cstddef>
#include <tensor.hpp>
#include <vector>
#include <cstdlib>

extern "C" {

typedef struct CTensor {
    void* handle;
} CTensor;

CTensor* create_tensor_py(const float* data, const size_t data_num_elems, const size_t* shape,
                          const size_t ndim);

void destroy_tensor_py(CTensor* tensor);

void get_tensor_shape_py(CTensor* tensor, const size_t** out_shape, size_t* out_ndim);

void get_add_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems);
}

CTensor* create_tensor_py(const float* data, const size_t data_num_elems, const size_t* shape,
                          const size_t ndim) {
    CTensor* out = (CTensor*)malloc(sizeof(CTensor));
    std::vector<std::size_t> shape_v{shape, shape + ndim};

    pml::Tensor<float>* tensor = new pml::Tensor<float>(data, data_num_elems, shape_v);

    out->handle = (void*)tensor;

    return out;
}

void destroy_tensor_py(CTensor* tensor) {
    pml::Tensor<float>* cpp_tensor = (pml::Tensor<float>*)tensor->handle;
    delete cpp_tensor;
}

void get_tensor_shape_py(CTensor* tensor, const size_t** out_shape, size_t* out_ndim) {
    pml::Tensor<float>* cpp_tensor = (pml::Tensor<float>*)tensor->handle;
    const std::vector<std::size_t>* shape_v = &cpp_tensor->get_shape();
    *out_shape = shape_v->data();
    *out_ndim = shape_v->size();
}

void get_add_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems) {
    auto* cpp_t1 = (pml::Tensor<float>*)t1->handle;
    auto* cpp_t2 = (pml::Tensor<float>*)t2->handle;

    pml::Tensor<float> result = pml::add(*cpp_t1, *cpp_t2);

    std::copy(result.get_data(), result.get_data() + result.get_data_num_elems(), out_data);
    *out_num_elems = result.get_data_num_elems();

    std::cout << result << std::endl;
}
