#ifndef TENSOR_HEADER_FILE
#define TENSOR_HEADER_FILE

#include <stddef.h>
#include <error_handling.h>
#include <container_type.h>
#include <dynarray.h>
#include <stdbool.h>

typedef struct tensor {
    container_type_t type;
    size_t n_dim;
    dynarray shape;
    dynarray strides;
    void* data;
    size_t data_num_elems;
    bool is_view;
    void (*print)(const struct tensor* self);
    struct tensor* (*view)(const struct tensor* self, const dynarray shape, pml_err_t* err);
    struct tensor* (*transpose)(const struct tensor* self, const int32_t idx1, const int32_t idx2, pml_err_t* err);
} tensor;

tensor* tensor_create(
    const void* data,
    const size_t data_len,
    const container_type_t type, 
    const size_t n_dimensions, 
    const dynarray shape, 
    pml_err_t* error
);

tensor* tensor_create_scalar(const void* value_ptr, const container_type_t type, pml_err_t* error);

tensor* tensor_create_zeros(
    const container_type_t type, 
    const size_t n_dimensions, 
    const dynarray shape, 
    pml_err_t* error
);

void tensor_free(tensor* obj);

bool tensor_shapes_broadcastable(tensor* left, tensor* right, pml_err_t* err);

tensor* tensor_add(tensor* left, tensor* right, container_type_t type, pml_err_t* err);

tensor* tensor_subtract(tensor* left, tensor* right, container_type_t type, pml_err_t* err);

tensor* tensor_multiply(tensor* left, tensor* right, container_type_t type, pml_err_t* err);

tensor* tensor_divide(tensor* left, tensor* right, container_type_t type, pml_err_t* err);

tensor* tensor_axis_sum(tensor* tensor, const size_t axis, pml_err_t* err);

tensor* tensor_axis_max(tensor* tensor, const size_t axis, pml_err_t* err);

tensor* tensor_axis_min(tensor* tensor, const size_t axis, pml_err_t* err);

tensor* tensor_axis_mean(tensor* tensor, const size_t axis, pml_err_t* err);

tensor* tensor_axis_var(tensor* tensor, const size_t axis, pml_err_t* err);

tensor* tensor_matmul(tensor* left, tensor* right, pml_err_t* err);

tensor* tensor_sum(const tensor* tensor, pml_err_t* err);

tensor* tensor_max(const tensor* tensor, pml_err_t* err);

tensor* tensor_min(const tensor* tensor, pml_err_t* err);

tensor* tensor_mean(const tensor* tensor, pml_err_t* err);

tensor* tensor_var(const tensor* tensor, pml_err_t* err);

tensor* tensor_sqrt(const tensor* tensor, pml_err_t* err);

tensor* tensor_log(const tensor* tensor, pml_err_t* err);

tensor* tensor_pow(const tensor* tensor, float exponent, pml_err_t* err);

typedef struct tensor_iterator {
    dynarray current_indices;
    void* data_ptr;
    size_t element_size;
    bool finished;
    bool started;
    void* (*get_next)(struct tensor_iterator* self, const dynarray* shape, const dynarray* strides, pml_err_t* err);
} tensor_iterator;

tensor_iterator* tensor_iterator_create(tensor* obj, pml_err_t* err);

void tensor_iterator_free(tensor_iterator* obj);

#endif // TENSOR_HEADER_FILE
