#include <storage.hpp>
#include <cstddef>
#include <tensor.hpp>
#include <operations.hpp>
#include <matmul.hpp>
#include <vector>
#include <cstdlib>

extern "C" {

typedef struct CTensor {
    void* handle;
} CTensor;

CTensor* create_tensor_py(float* data, const size_t data_num_elems, const size_t* shape,
                          const size_t ndim);

CTensor* create_tensor_scalar_py(const float scalar_value);

void destroy_tensor_py(CTensor* tensor);

void get_tensor_shape_py(CTensor* tensor, size_t* out_shape, size_t* out_ndim);

void get_add_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems);

void get_sum_operation_result(CTensor* operand, const size_t axis, const bool keep_dims,
                              float* out_data, size_t* out_num_elems);

void get_matmul_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems,
                                 size_t* out_n_dim);
}

CTensor* create_tensor_py(float* data, const size_t data_num_elems, const size_t* shape,
                          const size_t ndim) {
    CTensor* out = (CTensor*)malloc(sizeof(CTensor));
    std::vector<std::size_t> shape_v{shape, shape + ndim};

    std::span<float> data_span(data, data + data_num_elems);
    pml::Tensor<float>* tensor = new pml::Tensor<float>(data_span, shape_v);

    out->handle = (void*)tensor;

    return out;
}

void destroy_tensor_py(CTensor* tensor) {
    pml::Tensor<float>* cpp_tensor = static_cast<pml::Tensor<float>*>(tensor->handle);
    delete cpp_tensor;
}

void get_tensor_shape_py(CTensor* tensor, size_t* out_shape, size_t* out_ndim) {
    pml::Tensor<float>* cpp_tensor = static_cast<pml::Tensor<float>*>(tensor->handle);
    const std::vector<std::size_t>* shape_v = &cpp_tensor->get_shape();
    std::copy(shape_v->begin(), shape_v->end(), out_shape);
    *out_ndim = shape_v->size();
}

void get_add_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems) {
    auto* cpp_t1 = static_cast<pml::Tensor<float>*>(t1->handle);
    auto* cpp_t2 = static_cast<pml::Tensor<float>*>(t2->handle);

    pml::Tensor<float> result = pml::add(*cpp_t1, *cpp_t2);

    std::copy(result.get_data(), result.get_data() + result.get_data_num_elems(), out_data);
    *out_num_elems = result.get_data_num_elems();
}

void get_sum_operation_result(CTensor* operand, const size_t axis, const bool keep_dims,
                              float* out_data, size_t* out_num_elems) {
    auto* cpp_operand = static_cast<pml::Tensor<float>*>(operand->handle);

    pml::Tensor<float> result = pml::sum(*cpp_operand, axis, keep_dims);

    std::copy(result.get_data(), result.get_data() + result.get_data_num_elems(), out_data);
    *out_num_elems = result.get_data_num_elems();
}

void get_matmul_operation_result(CTensor* t1, CTensor* t2, float* out_data, size_t* out_num_elems,
                                 size_t* out_n_dim) {
    auto* cpp_t1 = static_cast<pml::Tensor<float>*>(t1->handle);
    auto* cpp_t2 = static_cast<pml::Tensor<float>*>(t2->handle);

    pml::Tensor<float> result = pml::matmul(*cpp_t1, *cpp_t2);

    std::copy(result.get_data(), result.get_data() + result.get_data_num_elems(), out_data);
    *out_num_elems = result.get_data_num_elems();
    *out_n_dim = result.ndim();
}

CTensor* create_tensor_scalar_py(const float scalar_value) {
    CTensor* out = (CTensor*)malloc(sizeof(CTensor));

    pml::Tensor<float>* tensor = new pml::Tensor<float>(scalar_value);

    out->handle = (void*)tensor;

    return out;
}
