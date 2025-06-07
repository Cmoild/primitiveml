#include <dynarray.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


static void dynarray_print(const dynarray* self);

static result_t dynarray_get_at(const dynarray* self, const size_t idx);

dynarray dynarray_create(const void *data, const size_t len, const container_type_t type, pml_err_t *error) {
    dynarray vector = {
        .get_at = dynarray_get_at,
        .print = dynarray_print,
    };
    switch (type) {
    case TYPE_INT32:
        // vector._container = data;
        vector._container = malloc(sizeof(int32_t) * len);
        if (!vector._container && len > 0) {
            *error = PML_OUT_OF_MEMORY;
            return vector;
        }
        memcpy(vector._container, data, len * sizeof(int32_t));
        vector._capacity = len;
        vector._size = len;
        vector.type = TYPE_INT32;
        *error = PML_OK;
        break;
    case TYPE_FLOAT:
        vector._container = malloc(sizeof(float) * len);
        if (!vector._container && len > 0) {
            *error = PML_OUT_OF_MEMORY;
            return vector;
        }
        memcpy(vector._container, data, len * sizeof(float));
        vector._capacity = len;
        vector._size = len;
        vector.type = TYPE_FLOAT;
        *error = PML_OK;
        break;
    default:
        *error = PML_WRONG_TYPE;
        break;
    }
    return vector;
}

static void dynarray_print(const dynarray* self) {
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
        return;
        break;
    }
    printf("[ ");
    for (size_t i = 0; i < self->_size; i++) {
        switch (self->type)
        {
        case TYPE_INT32:
            printf(fmt, *((int32_t*)self->_container + i));
            break;
        case TYPE_FLOAT:
            printf(fmt, *((float*)self->_container + i));
            break;
        }
    }
    printf("]\n");
}

static result_t dynarray_get_at(const dynarray* self, const size_t idx) {
    result_t res;
    if (idx >= self->_size || idx < 0) {
        res.err = PML_OUT_OF_BOUNDS;
        return res;
    }
    switch (self->type)
    {
    case TYPE_INT32:
        res.type = TYPE_INT32;
        res.val.i = *((int32_t*)self->_container + idx);
        res.err = PML_OK;
        break;
    case TYPE_FLOAT:
        res.type = TYPE_FLOAT;
        res.val.f = *((float*)self->_container + idx);
        res.err = PML_OK;
        break;
    default:
        res.err = PML_WRONG_TYPE;
        break;
    }
    return res;
}
