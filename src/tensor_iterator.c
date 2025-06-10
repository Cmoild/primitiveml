#include <tensor.h>
#include <dynarray.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>


static void* tensor_iterator_get_next(tensor_iterator* self, const dynarray* shape, const dynarray* strides, pml_err_t* err);

tensor_iterator* tensor_iterator_create(tensor* obj, pml_err_t* err) {
    tensor_iterator* iterator = (tensor_iterator*)malloc(sizeof(tensor_iterator));
    if (!iterator) {
        *err = PML_OUT_OF_MEMORY;
        return NULL;
    }
    switch (obj->type)
    {
    case TYPE_INT32:
        iterator->element_size = sizeof(int32_t);
        break;
    case TYPE_FLOAT:
        iterator->element_size = sizeof(float);
        break;
    default:
        printf("Type is not supported\n");
        free(iterator);
        return NULL;
        break;
    }
    iterator->finished = false;
    iterator->started = false;
    *err = PML_OK;
    iterator->get_next = tensor_iterator_get_next;
    iterator->data_ptr = obj->data;
    return iterator;
}

void tensor_iterator_free(tensor_iterator* obj) {
    dynarray_free(&obj->current_indices);
}

static void* tensor_iterator_get_next(tensor_iterator* self, const dynarray* shape, const dynarray* strides, pml_err_t* err) {
    if (self->finished) {
        *err = PML_OK;
        return NULL;
    }
    if (!self->started) {
        *err = PML_OK;
        self->current_indices = dynarray_zeros(shape->_size, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        self->started = true;
        return self->data_ptr;
    }
    bool counter_incremented = false;
    for (int64_t i = self->current_indices._size - 1; i >= 0; i--) {
        result_t dim_size = shape->get_at(shape, (size_t)i);
        result_t dim_counter = self->current_indices.get_at(&self->current_indices, (size_t)i);
        if (dim_size.err != PML_OK) {
            *err = dim_size.err;
            return NULL;
        }
        if (dim_counter.err != PML_OK) {
            *err = dim_counter.err;
            return NULL;
        }
        if (1 + dim_counter.val.i < dim_size.val.i) {
            int32_t new_value = 1 + dim_counter.val.i;
            self->current_indices.set_at(&self->current_indices, (size_t)i, &new_value);
            counter_incremented = true;
            break;
        }
        else {
            int32_t new_value = 0;
            self->current_indices.set_at(&self->current_indices, (size_t)i, &new_value);
        }
    }
    if (!counter_incremented) {
        *err = PML_OK;
        self->finished = true;
        return NULL;
    }
    size_t offset = 0;
    for (int64_t i = self->current_indices._size - 1; i >= 0; i--) {
        result_t dim_counter = self->current_indices.get_at(&self->current_indices, (size_t)i);
        if (dim_counter.err != PML_OK) {
            *err = dim_counter.err;
            return NULL;
        }
        result_t dim_stride = strides->get_at(strides, (size_t)i);
        if (dim_stride.err != PML_OK) {
            *err = dim_stride.err;
            return NULL;
        }
        offset += dim_stride.val.i * self->element_size * dim_counter.val.i;
    }
    return self->data_ptr + offset;
}
