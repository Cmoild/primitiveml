#include <tensor.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


static void tensor_print(const tensor* self);

tensor* tensor_create(
    const void* data,
    const size_t data_len,
    const container_type_t type, 
    const size_t n_dimentions, 
    const dynarray shape, 
    pml_err_t* error) {
    if (data_len == 0) {
        *error = PML_EMPTY_TENSOR;
        return NULL;
    }
    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = type;
    tnsr->shape = shape;
    tnsr->n_dim = n_dimentions;

    pml_err_t err_strides = PML_OK;
    dynarray strides = dynarray_create(NULL, 0, TYPE_INT32, &err_strides);
    if (err_strides != PML_OK) {
        *error = err_strides;
        return NULL;
    }
    if (n_dimentions != 0) {
        err_strides = strides.resize(&strides, n_dimentions);
        if (err_strides != PML_OK) {
            *error = err_strides;
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = n_dimentions - 1; i >= 0; i--) {
        strides.set_at(&strides, (size_t)i, (int32_t*)&n_elems);
        result_t res_get = shape.get_at(&shape, (size_t)i);
        if (res_get.err != PML_OK) {
            printf("Error (shape, res_get): %d, i: %d\n", res_get.err, i);
            *error = PML_INCORRECT_INPUT;
            return NULL;
        }
        n_elems *= (size_t)res_get.val.i;
    }
    
    switch (type) {
    case TYPE_INT32:
        tnsr->data = malloc(sizeof(int32_t) * n_elems);
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            return NULL;
        }
        memcpy(tnsr->data, data, n_elems * sizeof(int32_t));
        *error = PML_OK;
        break;
    case TYPE_FLOAT:
        tnsr->data = malloc(sizeof(float) * n_elems);
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            return NULL;
        }
        memcpy(tnsr->data, data, n_elems * sizeof(float));
        *error = PML_OK;
        break;
    default:
        *error = PML_WRONG_TYPE;
        printf("Type is not supported\n");
        return NULL;
        break;
    }
    tnsr->print = tensor_print;
    tnsr->data_num_elems = n_elems;
    tnsr->strides = strides;
    return tnsr;
}

static void tensor_print(const tensor* self) {
    char* fmt;
    switch (self->type)
    {
    case TYPE_INT32:
        fmt = "%i ";
        break;
    case TYPE_FLOAT:
        fmt = "%f ";
        break;
    default:
        printf("Type is not supported\n");
        return;
        break;
    }
    printf("Data: [ ");
    for (size_t i = 0; i < self->data_num_elems; i++) {
        switch (self->type)
        {
        case TYPE_INT32:
            printf(fmt, *((int32_t*)self->data + i));
            break;
        case TYPE_FLOAT:
            printf(fmt, *((float*)self->data + i));
            break;
        }
    }
    printf("]\n");
    printf("Shape: ");
    self->shape.print(&self->shape);
    printf("Strides: ");
    self->strides.print(&self->strides);
}
