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
    void (*print)(const struct tensor* self);
} tensor;

tensor* tensor_create(
    const void* data,
    const size_t data_len,
    const container_type_t type, 
    const size_t n_dimentions, 
    const dynarray shape, 
    pml_err_t* error
);

void tensor_free(tensor* obj);

bool tensor_shapes_broadcastable(tensor* left, tensor* right, pml_err_t* err);

#endif // TENSOR_HEADER_FILE
