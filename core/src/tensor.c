#include <tensor.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <tensor_index.h>

#define USE_BLAS
#define USE_PARALLEL_ITERATOR
#ifdef USE_BLAS
#include <cblas.h>
#endif

static void tensor_print(const tensor* self);

static tensor* tensor_view(const tensor* self, const dynarray shape, pml_err_t* err);

static tensor* tensor_transpose(const tensor* self, const int32_t idx1, const int32_t idx2, pml_err_t* err);

static tensor* tensor_unsqueeze(const tensor* self, const int32_t idx, pml_err_t* err);

static tensor* tensor_slice(const tensor* self, const index_tuple_t slices, pml_err_t* err);

static tensor* tensor_contiguous(const tensor* self, pml_err_t* err);

tensor* tensor_create(
    const void* data,
    const size_t data_len,
    const container_type_t type, 
    const size_t n_dimensions, 
    const dynarray shape, 
    pml_err_t* error) {
    
    if (data_len == 0) {
        *error = PML_EMPTY_TENSOR;
        return NULL;
    }
    // if (shape._size == 0) {
    //     *error = PML_INCORRECT_INPUT;
    //     return NULL;
    // }
    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = type;
    tnsr->shape = shape;
    tnsr->n_dim = n_dimensions;

    pml_err_t err_strides = PML_OK;
    dynarray strides = dynarray_create(NULL, 0, TYPE_INT32, &err_strides);
    if (err_strides != PML_OK) {
        *error = err_strides;
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    if (n_dimensions != 0) {
        err_strides = strides.resize(&strides, n_dimensions);
        if (err_strides != PML_OK) {
            *error = err_strides;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = n_dimensions - 1; i >= 0; i--) {
        strides.set_at(&strides, (size_t)i, (int32_t*)&n_elems);
        result_t res_get = shape.get_at(&shape, (size_t)i);
        if (res_get.err != PML_OK) {
            *error = PML_INCORRECT_INPUT;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
        n_elems *= (size_t)res_get.val.i;
    }
    
    switch (type) {
    case TYPE_INT32:
        tnsr->data = malloc(sizeof(int32_t) * n_elems);
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            dynarray_free(&strides);
            return NULL;
        }
        memcpy(tnsr->data, data, n_elems * sizeof(int32_t));
        *error = PML_OK;
        break;
    case TYPE_FLOAT:
        tnsr->data = malloc(sizeof(float) * n_elems);
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            dynarray_free(&strides);
            return NULL;
        }
        memcpy(tnsr->data, data, n_elems * sizeof(float));
        *error = PML_OK;
        break;
    default:
        *error = PML_WRONG_TYPE;
        free(tnsr);
        dynarray_free(&strides);
        printf("Type is not supported\n");
        return NULL;
        break;
    }
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->slice = tensor_slice;
    tnsr->contiguous = tensor_contiguous;
    tnsr->data_num_elems = n_elems;
    tnsr->strides = strides;
    tnsr->is_view = false;
    return tnsr;
}

tensor* tensor_create_scalar(const void* value_ptr, const container_type_t type, pml_err_t* error) {
    if (!value_ptr) {
        *error = PML_EMPTY_TENSOR;
        return NULL;
    }
    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = type;
    tnsr->n_dim = 0;
    tnsr->data_num_elems = 1;
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->contiguous = tensor_contiguous;
    tnsr->slice = tensor_slice;
    switch (type) {
    case TYPE_INT32:
        tnsr->data = malloc(sizeof(int32_t));
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            return NULL;
        }
        memcpy(tnsr->data, value_ptr, sizeof(int32_t));
        *error = PML_OK;
        break;
    case TYPE_FLOAT:
        tnsr->data = malloc(sizeof(float));
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            return NULL;
        }
        memcpy(tnsr->data, value_ptr, sizeof(float));
        *error = PML_OK;
        break;
    default:
        *error = PML_WRONG_TYPE;
        free(tnsr);
        printf("Type is not supported\n");
        return NULL;
        break;
    }
    *error = PML_OK;
    dynarray strides = dynarray_create(NULL, 0, TYPE_INT32, error);
    if (*error != PML_OK) {
        free(tnsr);
        return NULL;
    }
    dynarray shape = dynarray_create(NULL, 0, TYPE_INT32, error);
    if (*error != PML_OK) {
        free(tnsr);
        return NULL;
    }
    tnsr->shape = shape;
    tnsr->strides = strides;
    tnsr->is_view = false;
    return tnsr;
}

tensor* tensor_create_zeros(
    const container_type_t type, 
    const size_t n_dimensions, 
    const dynarray shape, 
    pml_err_t* error) {
    
    // if (shape._size == 0) {
    //     *error = PML_INCORRECT_INPUT;
    //     return NULL;
    // }
    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = type;
    tnsr->shape = shape;
    tnsr->n_dim = n_dimensions;

    pml_err_t err_strides = PML_OK;
    dynarray strides = dynarray_create(NULL, 0, TYPE_INT32, &err_strides);
    if (err_strides != PML_OK) {
        *error = err_strides;
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    if (n_dimensions != 0) {
        err_strides = strides.resize(&strides, n_dimensions);
        if (err_strides != PML_OK) {
            *error = err_strides;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = n_dimensions - 1; i >= 0; i--) {
        strides.set_at(&strides, (size_t)i, (int32_t*)&n_elems);
        result_t res_get = shape.get_at(&shape, (size_t)i);
        if (res_get.err != PML_OK) {
            *error = PML_INCORRECT_INPUT;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
        n_elems *= (size_t)res_get.val.i;
    }
    if (n_elems == 0) {
        *error = PML_EMPTY_TENSOR;
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    switch (type) {
    case TYPE_INT32:
        tnsr->data = calloc(n_elems, sizeof(int32_t));
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            dynarray_free(&strides);
            return NULL;
        }
        *error = PML_OK;
        break;
    case TYPE_FLOAT:
        tnsr->data = calloc(n_elems, sizeof(float));
        if (!tnsr->data) {
            *error = PML_OUT_OF_MEMORY;
            free(tnsr);
            dynarray_free(&strides);
            return NULL;
        }
        *error = PML_OK;
        break;
    default:
        *error = PML_WRONG_TYPE;
        free(tnsr);
        dynarray_free(&strides);
        printf("Type is not supported\n");
        return NULL;
        break;
    }
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->contiguous = tensor_contiguous;
    tnsr->slice = tensor_slice;
    tnsr->data_num_elems = n_elems;
    tnsr->strides = strides;
    tnsr->is_view = false;
    return tnsr;
}

static tensor* tensor_view(const tensor* self, const dynarray shape, pml_err_t* err) {
    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = self->type;
    tnsr->n_dim = shape._size;

    pml_err_t err_strides = PML_OK;
    dynarray strides = dynarray_create(NULL, 0, TYPE_INT32, &err_strides);
    if (err_strides != PML_OK) {
        *err = err_strides;
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    if (tnsr->n_dim != 0) {
        err_strides = strides.resize(&strides, tnsr->n_dim);
        if (err_strides != PML_OK) {
            *err = err_strides;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = (int)tnsr->n_dim - 1; i >= 0; i--) {
        strides.set_at(&strides, (size_t)i, (int32_t*)&n_elems);
        result_t res_get = shape.get_at(&shape, (size_t)i);
        if (res_get.err != PML_OK) {
            *err = PML_INCORRECT_INPUT;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
        n_elems *= (size_t)res_get.val.i;
    }
    if (n_elems == 0) {
        *err = PML_EMPTY_TENSOR;
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    tnsr->shape = shape;
    tnsr->data = self->data;
    tnsr->data_num_elems = n_elems;
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->slice = tensor_slice;
    tnsr->contiguous = tensor_contiguous;
    tnsr->strides = strides;
    tnsr->is_view = true;
    return tnsr;
}

static tensor* tensor_slice(const tensor* self, const index_tuple_t slices, pml_err_t* err) {
    if (slices.len > self->n_dim || self->n_dim == 0) {
        *err = PML_OUT_OF_BOUNDS;
        return NULL;
    }
    char* new_data_ptr = self->data;
    size_t element_size;
    switch (self->type)
    {
    case TYPE_INT32:
        element_size = sizeof(int32_t);
        break;
    case TYPE_FLOAT:
        element_size = sizeof(float);
        break;
    default:
        printf("Type is not supported\n");
        *err = PML_WRONG_TYPE;
        return NULL;
        break;
    }
    size_t offset = 0;
    for (size_t i = 0; i < slices.len; i++) {
        tensor_index_t cur_slice = slices.items[i];
        int32_t cur_stride = self->strides.get_at(&self->strides, i).val.i;
        int32_t cur_shape = self->shape.get_at(&self->shape, i).val.i;
        switch (cur_slice.type)
        {
        case IDX_SLICE:
            if ((cur_slice.value.slice.start >= cur_shape) || \
                (cur_slice.value.slice.start < 0 && cur_shape + cur_slice.value.slice.start < 0) || \
                (cur_slice.value.slice.end > cur_shape) || \
                (cur_slice.value.slice.end < 0 && cur_shape + cur_slice.value.slice.end < 0)) {
                *err = PML_OUT_OF_BOUNDS;
                return NULL;
            }
            if (cur_slice.value.slice.start < 0) {
                cur_slice.value.slice.start = cur_shape + cur_slice.value.slice.start;
            }
            if (cur_slice.value.slice.end < 0) {
                cur_slice.value.slice.end = cur_shape + cur_slice.value.slice.end;
            }
            if (cur_slice.value.slice.end - cur_slice.value.slice.start < 0) {
                *err = PML_INCORRECT_INPUT;
                return NULL;
            } else if (cur_slice.value.slice.end - cur_slice.value.slice.start == 0) {
                *err = PML_EMPTY_TENSOR;
                return NULL;
            }
            offset += cur_stride * element_size * cur_slice.value.slice.start;
            break;
        case IDX_INT:
            if ((cur_slice.value.index < 0 && cur_shape + cur_slice.value.index < 0) || \
                (cur_slice.value.index >= cur_shape)) {
                *err = PML_OUT_OF_BOUNDS;
                return NULL;
            }
            if (cur_slice.value.index < 0) {
                cur_slice.value.index = cur_shape + cur_slice.value.index;
            }
            offset += cur_stride * element_size * cur_slice.value.index;
            break;
        default:
            *err = PML_WRONG_TYPE;
            return NULL;
            break;
        }
    }
    dynarray new_shape = dynarray_clone(&self->shape, err);
    if (*err != PML_OK) {
        return NULL;
    }
    dynarray new_strides = dynarray_clone(&self->strides, err);
    if (*err != PML_OK) {
        dynarray_free(&new_shape);
        return NULL;
    }
    size_t out_n_dim = self->n_dim;
    for (int i = (int)slices.len - 1; i >= 0; i--) {
        tensor_index_t cur_slice = slices.items[i];
        if (cur_slice.type == IDX_INT) {
            new_shape.delete_at(&new_shape, (size_t)i);
            new_strides.delete_at(&new_strides, (size_t)i);
            out_n_dim--;
        } else if (cur_slice.type == IDX_SLICE) {
            int32_t cur_new_shape = cur_slice.value.slice.end - cur_slice.value.slice.start;
            new_shape.set_at(&new_shape, i, &cur_new_shape);
        }
    }
    size_t out_num_elems = 1;
    for (size_t i = 0; i < new_shape._size; i++) {
        out_num_elems *= (size_t)new_shape.get_at(&new_shape, i).val.i;
    }
    tensor* out = (tensor*)malloc(sizeof(tensor));
    if (!out) {
        dynarray_free(&new_shape);
        dynarray_free(&new_strides);
        *err = PML_OUT_OF_MEMORY;
        return NULL;
    }
    out->data = new_data_ptr + offset;
    out->data_num_elems = out_num_elems;
    out->is_view = true;
    out->n_dim = out_n_dim;
    out->print = tensor_print;
    out->shape = new_shape;
    out->strides = new_strides;
    out->transpose = tensor_transpose;
    out->type = self->type;
    out->unsqueeze = tensor_unsqueeze;
    out->view = tensor_view;
    out->slice = tensor_slice;
    out->contiguous = tensor_contiguous;

    return out;
}

static tensor* tensor_transpose(const tensor* self, const int32_t idx1, const int32_t idx2, pml_err_t* err) {
    int32_t idx_left = idx1;
    int32_t idx_right = idx2;
    if (idx_left < 0) {
        idx_left = (int32_t)self->n_dim + idx_left;
    }
    if (idx_right < 0) {
        idx_right = (int32_t)self->n_dim + idx_right;
    }
    if (idx_left < 0 || idx_left >= (int32_t)self->n_dim) {
        *err = PML_OUT_OF_BOUNDS;
        return NULL;
    }
    if (idx_right < 0 || idx_right >= (int32_t)self->n_dim) {
        *err = PML_OUT_OF_BOUNDS;
        return NULL;
    }

    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = self->type;
    tnsr->n_dim = self->n_dim;
    tnsr->shape = dynarray_clone(&self->shape, err);
    if (*err != PML_OK) {
        printf("Error: %d\n", *err);
        free(tnsr);
        return NULL;
    }
    tnsr->data = self->data;
    tnsr->data_num_elems = self->data_num_elems;
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->slice = tensor_slice;
    tnsr->contiguous = tensor_contiguous;
    tnsr->strides = dynarray_clone(&self->strides, err);
    if (*err != PML_OK) {
        dynarray_free(&tnsr->shape);
        free(tnsr);
        return NULL;
    }
    tnsr->is_view = true;

    int32_t shape_left_swapped, shape_right_swapped;
    shape_left_swapped = self->shape.get_at(&self->shape, (size_t)idx_right).val.i;
    shape_right_swapped = self->shape.get_at(&self->shape, (size_t)idx_left).val.i;
    tnsr->shape.set_at(&tnsr->shape, (size_t)idx_left, &shape_left_swapped);
    tnsr->shape.set_at(&tnsr->shape, (size_t)idx_right, &shape_right_swapped);

    int32_t stride_left_swapped, stride_right_swapped;
    stride_left_swapped = self->strides.get_at(&self->strides, (size_t)idx_right).val.i;
    stride_right_swapped = self->strides.get_at(&self->strides, (size_t)idx_left).val.i;
    tnsr->strides.set_at(&tnsr->strides, (size_t)idx_left, &stride_left_swapped);
    tnsr->strides.set_at(&tnsr->strides, (size_t)idx_right, &stride_right_swapped);

    return tnsr;
}

static tensor* tensor_unsqueeze(const tensor* self, const int32_t idx, pml_err_t* err) {
    int32_t positive_idx = idx;
    if (idx < 0) {
        positive_idx = (int32_t)self->n_dim + idx + 1;
    }
    if (positive_idx < 0 || positive_idx > (int32_t)self->n_dim) {
        *err = PML_OUT_OF_BOUNDS;
        return NULL;
    }

    tensor* tnsr = (tensor*)malloc(sizeof(tensor));
    tnsr->type = self->type;
    tnsr->n_dim = self->n_dim + 1;
    tnsr->shape = dynarray_clone(&self->shape, err);
    if (*err != PML_OK) {
        printf("Error: %d\n", *err);
        free(tnsr);
        return NULL;
    }
    tnsr->data = self->data;
    tnsr->data_num_elems = self->data_num_elems;
    tnsr->print = tensor_print;
    tnsr->view = tensor_view;
    tnsr->transpose = tensor_transpose;
    tnsr->unsqueeze = tensor_unsqueeze;
    tnsr->slice = tensor_slice;
    tnsr->contiguous = tensor_contiguous;
    tnsr->strides = dynarray_clone(&self->strides, err);
    if (*err != PML_OK) {
        dynarray_free(&tnsr->shape);
        free(tnsr);
        return NULL;
    }
    tnsr->is_view = true;

    int32_t shape_at_idx, stride_at_idx;
    if (positive_idx < (int32_t)tnsr->shape._size) {
        shape_at_idx = tnsr->shape.get_at(&tnsr->shape, (size_t)positive_idx).val.i;
        stride_at_idx = tnsr->strides.get_at(&tnsr->strides, (size_t)positive_idx).val.i;
    } else {
        shape_at_idx = 1;
        stride_at_idx = 1;
    }

    int32_t new_shape = 1;
    int32_t new_stride = shape_at_idx * stride_at_idx;
    tnsr->shape.insert_at(&tnsr->shape, (size_t)positive_idx, &new_shape);
    tnsr->strides.insert_at(&tnsr->strides, (size_t)positive_idx, &new_stride);

    return tnsr;
}

static tensor* tensor_contiguous(const tensor* self, pml_err_t* err) {
    // TODO
}

static void print_helper(const tensor* self, size_t shape_idx, void* data) {
    result_t dim_size = self->shape.get_at(&self->shape, shape_idx);
    if (dim_size.err != PML_OK) {
        return;
    }
    if (shape_idx == self->shape._size - 1) {
        printf("[ ");
        result_t stride = self->strides.get_at(&self->strides, shape_idx);
        if (stride.err != PML_OK) {
            return;
        }
        for (size_t i = 0; i < (size_t)dim_size.val.i; i++) {
            switch (self->type)
            {
            case TYPE_INT32:
                int32_t ival = *((int32_t*)data + i * stride.val.i);
                printf("%d ", ival);
                break;
            case TYPE_FLOAT:
                float fval = *((float*)data + i * stride.val.i);
                printf("%f ", fval);
                break;
            default:
                printf("Type is not supported\n");
                return;
                break;
            }
        }
        printf("]");
    }
    else {
        printf("[");
        result_t stride = self->strides.get_at(&self->strides, shape_idx);
        if (stride.err != PML_OK) {
            return;
        }
        for (size_t i = 0; i < (size_t)dim_size.val.i; i++) {
            void* sub_data;
            switch (self->type)
            {
            case TYPE_INT32:
                sub_data = (int32_t*)data + i * stride.val.i;
                break;
            case TYPE_FLOAT:
                sub_data = (float*)data + i * stride.val.i;
                break;
            default:
                printf("Type is not supported\n");
                return;
                break;
            }
            print_helper(self, shape_idx + 1, sub_data);
            if (i < (size_t)(dim_size.val.i - 1)) {
                printf(",\n");
            }
        }
        printf("]");
    }
}

static void tensor_print(const tensor* self) {
    if (self->n_dim != 0) {
        printf("Tensor: {\n");
        print_helper(self, 0, self->data);
        printf("\n");
        printf("Shape: ");
        self->shape.print(&self->shape);
        printf("Strides: ");
        self->strides.print(&self->strides);
        printf("}\n");
    }
    else {
        switch (self->type)
        {
        case TYPE_INT32:
            printf("Tensor (scalar): %d\n", *(int32_t*)self->data);
            break;
        case TYPE_FLOAT:
            printf("Tensor (scalar): %f\n", *(float*)self->data);
            break;
        default:
            printf("Type is not supported\n");
            return;
            break;
        }
    }
}

void tensor_free(tensor* obj) {
    dynarray_free(&obj->shape);
    dynarray_free(&obj->strides);
    if (obj->data && !obj->is_view) {
        free(obj->data);
    }
}

bool tensor_shapes_broadcastable(tensor* left, tensor* right, pml_err_t* err) {
    for (int i = 0; i < (size_t)left->shape._size && i < (size_t)right->shape._size; i++) {
        result_t left_cur = left->shape.get_at(&left->shape, left->shape._size - 1 - i); 
        if (left_cur.err != PML_OK) {
            *err = left_cur.err;
            return 0;
        }
        result_t right_cur = right->shape.get_at(&right->shape, right->shape._size - 1 - i);
        if (right_cur.err != PML_OK) {
            *err = right_cur.err;
            return 0;
        }
        if (left_cur.val.i == right_cur.val.i) {
            continue;
        }
        if (left_cur.val.i == 1 || right_cur.val.i == 1) {
            continue;
        }
        return 0;
    }

    return 1;
}

typedef struct tensor_broadcast_tensors_tuple_t {
    dynarray left_strides;
    dynarray right_strides;
    dynarray result_shape;
} tensor_broadcast_tensors_tuple_t;

static tensor_broadcast_tensors_tuple_t
tensor_broadcast_tensors(const tensor* left, const tensor* right, pml_err_t* err) {
    tensor_broadcast_tensors_tuple_t out;
    size_t max_n_dim = (left->n_dim > right->n_dim) ? left->n_dim : right->n_dim;
    dynarray result_shape = dynarray_zeros(max_n_dim, TYPE_INT32, err);
    if (*err != PML_OK) {
        return out;
    }
    dynarray left_strides = dynarray_zeros(max_n_dim, TYPE_INT32, err);
    if (*err != PML_OK) {
        return out;
    }
    dynarray right_strides = dynarray_zeros(max_n_dim, TYPE_INT32, err);
    if (*err != PML_OK) {
        return out;
    }
    // Tensors must be broadcastable at this moment (check it before calling this function)
    for (int i = 0; i < (int)max_n_dim; i++) {
        if ((size_t)i < left->shape._size && (size_t)i < right->shape._size) {
            result_t left_cur = left->shape.get_at(&left->shape, left->shape._size - 1 - i); 
            if (left_cur.err != PML_OK) {
                *err = left_cur.err;
                return out;
            }
            result_t right_cur = right->shape.get_at(&right->shape, right->shape._size - 1 - i);
            if (right_cur.err != PML_OK) {
                *err = right_cur.err;
                return out;
            }
            result_t left_stride_cur = left->strides.get_at(&left->strides, left->strides._size - 1 - i);
            if (left_stride_cur.err != PML_OK) {
                *err = left_stride_cur.err;
                return out;
            }
            result_t right_stride_cur = right->strides.get_at(&right->strides, right->strides._size - 1 - i);
            if (right_stride_cur.err != PML_OK) {
                *err = right_stride_cur.err;
                return out;
            }
            if (left_cur.val.i == right_cur.val.i) {
                left_strides.set_at(&left_strides, left_strides._size - 1 - i, &left_stride_cur.val.i);
                right_strides.set_at(&right_strides, right_strides._size - 1 - i, &right_stride_cur.val.i);
            } else if (left_cur.val.i != right_cur.val.i && left_cur.val.i == 1) {
                int32_t new_val = 0;
                left_strides.set_at(&left_strides, left_strides._size - 1 - i, &new_val);
                right_strides.set_at(&right_strides, right_strides._size - 1 - i, &right_stride_cur.val.i);
            } else if (left_cur.val.i != right_cur.val.i && right_cur.val.i == 1) {
                left_strides.set_at(&left_strides, left_strides._size - 1 - i, &left_stride_cur.val.i);
                int32_t new_val = 0;
                right_strides.set_at(&right_strides, right_strides._size - 1 - i, &new_val);
            }
            int32_t max_dim_size = (left_cur.val.i > right_cur.val.i) ? left_cur.val.i : right_cur.val.i;
            result_shape.set_at(&result_shape, result_shape._size - 1 - i, &max_dim_size);
        } else if ((size_t)i < left->shape._size && (size_t)i >= right->shape._size) {
            result_t left_cur = left->shape.get_at(&left->shape, left->shape._size - 1 - i); 
            if (left_cur.err != PML_OK) {
                *err = left_cur.err;
                return out;
            }
            result_t left_stride_cur = left->strides.get_at(&left->strides, left->strides._size - 1 - i);
            if (left_stride_cur.err != PML_OK) {
                *err = left_stride_cur.err;
                return out;
            }
            int32_t new_val = 0;
            left_strides.set_at(&left_strides, left_strides._size - 1 - i, &left_stride_cur.val.i);
            right_strides.set_at(&right_strides, right_strides._size - 1 - i, &new_val);
            result_shape.set_at(&result_shape, result_shape._size - 1 - i, &left_cur.val.i);
        } else if ((size_t)i >= left->shape._size && (size_t)i < right->shape._size) {
            result_t right_cur = right->shape.get_at(&right->shape, right->shape._size - 1 - i);
            if (right_cur.err != PML_OK) {
                *err = right_cur.err;
                return out;
            }
            result_t right_stride_cur = right->strides.get_at(&right->strides, right->strides._size - 1 - i);
            if (right_stride_cur.err != PML_OK) {
                *err = right_stride_cur.err;
                return out;
            }
            int32_t new_val = 0;
            left_strides.set_at(&left_strides, left_strides._size - 1 - i, &new_val);
            right_strides.set_at(&right_strides, right_strides._size - 1 - i, &right_stride_cur.val.i);
            result_shape.set_at(&result_shape, result_shape._size - 1 - i, &right_cur.val.i);
        }
    }
    out.left_strides = left_strides;
    out.right_strides = right_strides;
    out.result_shape = result_shape;
    return out;
}

static tensor* tensor_apply_elementwise_operation(
    tensor* left, tensor* right, pml_err_t* err, container_type_t type,
    void (*operation)(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err)
) {
    *err = PML_OK;
    tensor* result;
    if (tensor_shapes_broadcastable(left, right, err)) {
        tensor_broadcast_tensors_tuple_t out = tensor_broadcast_tensors(left, right, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            return NULL;
        }
        result = tensor_create_zeros(type, out.result_shape._size, out.result_shape, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            return NULL;
        }
        tensor_iterator* iterator_left = tensor_iterator_create(left, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            free(iterator_left);
            return NULL;
        }
        tensor_iterator* iterator_right = tensor_iterator_create(right, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            free(iterator_left);
            free(iterator_right);
            return NULL;
        }
        tensor_iterator* iterator_result = tensor_iterator_create(result, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            free(iterator_left);
            free(iterator_right);
            free(iterator_result);
            return NULL;
        }
        #ifndef USE_PARALLEL_ITERATOR
        do {
            void* left_ptr = iterator_left->get_next(iterator_left, &out.result_shape, &out.left_strides, err);
            void* right_ptr = iterator_right->get_next(iterator_right, &out.result_shape, &out.right_strides, err);
            void* result_ptr = iterator_result->get_next(iterator_result, &out.result_shape, &result->strides, err);
            if (left_ptr && right_ptr && result_ptr) {
                operation(result_ptr, left_ptr, right_ptr, type, err);
                if (*err != PML_OK) {
                    tensor_free(result);
                    free(result);
                    break;
                }
                // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
            }
        } while (!iterator_left->finished && !iterator_right->finished && !iterator_result->finished);
        #else
        bool not_ok = false;
        #pragma omp parallel for
        for (size_t i = 0; i < result->data_num_elems; i++) {
            void* left_ptr = iterator_left->get_by_idx(iterator_left, &out.result_shape, &out.left_strides, i, err);
            void* right_ptr = iterator_right->get_by_idx(iterator_right, &out.result_shape, &out.right_strides, i, err);
            void* result_ptr = iterator_result->get_by_idx(iterator_result, &out.result_shape, &result->strides, i, err);
            if (left_ptr && right_ptr && result_ptr) {
                operation(result_ptr, left_ptr, right_ptr, type, err);
                if (*err != PML_OK) {
                    not_ok = true;
                }
            }
        }
        if (not_ok) {
            tensor_free(result);
            free(result);
        }
        #endif
        dynarray_free(&out.left_strides);
        dynarray_free(&out.right_strides);
        tensor_iterator_free(iterator_left);
        free(iterator_left);
        tensor_iterator_free(iterator_right);
        free(iterator_right);
        tensor_iterator_free(iterator_result);
        free(iterator_result);
    } else {
        if (*err == PML_OK) {
            *err = PML_TENSORS_NOT_BROADCASTABLE;
        }
        return NULL;
    }
    return result;
}

static void tensor_add_operation(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = *(int32_t*)left_ptr + *(int32_t*)right_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = *(float*)left_ptr + *(float*)right_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_add(tensor* left, tensor* right, container_type_t type, pml_err_t* err) {
    return tensor_apply_elementwise_operation(left, right, err, type, tensor_add_operation);
}

static void tensor_max_binary_operation(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = (*(int32_t*)left_ptr > *(int32_t*)right_ptr) ? *(int32_t*)left_ptr : *(int32_t*)right_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = (*(float*)left_ptr > *(float*)right_ptr) ? *(float*)left_ptr : *(float*)right_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_max_binary(tensor* left, tensor* right, pml_err_t* err) {
    return tensor_apply_elementwise_operation(left, right, err, left->type, tensor_max_binary_operation);
}

static void tensor_subtract_operation(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = *(int32_t*)left_ptr - *(int32_t*)right_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = *(float*)left_ptr - *(float*)right_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_subtract(tensor* left, tensor* right, container_type_t type, pml_err_t* err) {
    return tensor_apply_elementwise_operation(left, right, err, type, tensor_subtract_operation);
}

static void tensor_multiply_operation(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = *(int32_t*)left_ptr * *(int32_t*)right_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = *(float*)left_ptr * *(float*)right_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_multiply(tensor* left, tensor* right, container_type_t type, pml_err_t* err) {
    return tensor_apply_elementwise_operation(left, right, err, type, tensor_multiply_operation);
}

static void tensor_divide_operation(void* result_ptr, void* left_ptr, void* right_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = *(int32_t*)left_ptr / *(int32_t*)right_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = *(float*)left_ptr / *(float*)right_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_divide(tensor* left, tensor* right, container_type_t type, pml_err_t* err) {
    return tensor_apply_elementwise_operation(left, right, err, type, tensor_divide_operation);
}

static tensor* tensor_apply_unary_axis(
    tensor* obj, const size_t axis, pml_err_t* err, container_type_t type,
    void (*axis_operation)(
        tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
        const dynarray* strides_res, const dynarray* transposed_obj_strides,
        const dynarray* res_shape, const dynarray* transposed_obj_shape,
        const size_t dim_size,
        container_type_t type, pml_err_t* err
    )
) {
    if (obj->n_dim == 0) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    *err = PML_OK;
    dynarray transposed_shape = dynarray_clone(&obj->shape, err);
    if (*err != PML_OK) {
        return NULL;
    }
    if (axis >= transposed_shape._size) {
        *err = PML_OUT_OF_BOUNDS;
        dynarray_free(&transposed_shape);
        return NULL;
    }
    dynarray transposed_strides = dynarray_clone(&obj->strides, err);
    if (*err != PML_OK) {
        dynarray_free(&transposed_shape);
        return NULL;
    }
    int32_t* raw_transposed_shape = (int32_t*)transposed_shape._container;
    int32_t* raw_transposed_strides = (int32_t*)transposed_strides._container;
    int32_t shape_at_axis = raw_transposed_shape[axis];
    int32_t stride_at_axis = raw_transposed_strides[axis];
    int idx = 0;
    for (int i = 0; i < (int)transposed_shape._size - 1; i++) {
        if (i != (int)axis) {
            raw_transposed_shape[i] = raw_transposed_shape[idx];
            raw_transposed_strides[i] = raw_transposed_strides[idx];
            idx++;
        } else {
            raw_transposed_shape[i] = raw_transposed_shape[idx + 1];
            raw_transposed_strides[i] = raw_transposed_strides[idx + 1];
            idx += 2;
        }
    }
    raw_transposed_shape[transposed_shape._size - 1] = shape_at_axis;
    raw_transposed_strides[transposed_shape._size - 1] = stride_at_axis;

    dynarray output_shape = dynarray_create(raw_transposed_shape, transposed_shape._size - 1, TYPE_INT32, err);
    if (*err != PML_OK) {
        dynarray_free(&transposed_shape);
        dynarray_free(&transposed_strides);
        return NULL;
    }
    tensor* result = tensor_create_zeros(type, transposed_shape._size - 1, output_shape, err);
    if (transposed_shape._size - 1 == 0) {
        // tensor_create_zeros throws error when n_dim == 0
        int64_t zero = 0;
        result = tensor_create_scalar(&zero, type, err);
    }
    if (*err != PML_OK) {
        dynarray_free(&transposed_shape);
        dynarray_free(&transposed_strides);
        return NULL;
    }
    tensor_iterator* iterator_result = tensor_iterator_create(result, err);
    if (*err != PML_OK) {
        dynarray_free(&transposed_shape);
        dynarray_free(&transposed_strides);
        tensor_free(result);
        free(result);
        free(iterator_result);
        return NULL;
    }
    tensor_iterator* iterator_obj = tensor_iterator_create(obj, err);
    if (*err != PML_OK) {
        dynarray_free(&transposed_shape);
        dynarray_free(&transposed_strides);
        tensor_free(result);
        free(result);
        tensor_iterator_free(iterator_result);
        free(iterator_result);
        free(iterator_obj);
        return NULL;
    }
    axis_operation(
        iterator_result, iterator_obj,
        &result->strides, &transposed_strides,
        &result->shape, &transposed_shape,
        (size_t)shape_at_axis, type, err
    );
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        return NULL;
    }
    dynarray_free(&transposed_shape);
    dynarray_free(&transposed_strides);
    tensor_iterator_free(iterator_result);
    free(iterator_result);
    tensor_iterator_free(iterator_obj);
    free(iterator_obj);
    return result;
}

static void axis_operation_sum_util(void* result_ptr, void* obj_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i += *(int32_t*)obj_ptr;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f += *(float*)obj_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_sum(
    tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
    const dynarray* strides_res, const dynarray* transposed_obj_strides,
    const dynarray* res_shape, const dynarray* transposed_obj_shape,
    const size_t dim_size,
    container_type_t type, pml_err_t* err
) {
    do {
        void* result_ptr = iterator_res->get_next(iterator_res, res_shape, strides_res, err);
        for (size_t i = 0; i < dim_size && result_ptr; i++) {
            void* obj_ptr = iterator_obj->get_next(iterator_obj, transposed_obj_shape, transposed_obj_strides, err);
            if (obj_ptr && *err == PML_OK) {
                axis_operation_sum_util(result_ptr, obj_ptr, type, err);
                if (*err != PML_OK) {
                    return;
                }
            } else {
                return;
            }
        }
    } while (!iterator_res->finished && !iterator_obj->finished);
}

tensor* tensor_axis_sum(tensor* tensor, const size_t axis, pml_err_t* err) {
    return tensor_apply_unary_axis(tensor, axis, err, tensor->type, axis_operation_sum);
}

static void axis_operation_max_util(void* result_ptr, void* obj_ptr, const size_t iter, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        if (iter == 0) {
            int32_t* result_ptr_i = result_ptr;
            *result_ptr_i = *(int32_t*)obj_ptr;
        } else {
            int32_t* result_ptr_i = result_ptr;
            *result_ptr_i = (*result_ptr_i > *(int32_t*)obj_ptr) ? *result_ptr_i : *(int32_t*)obj_ptr;
        }
        break;
    case TYPE_FLOAT:
        if (iter == 0) {
            float* result_ptr_i = result_ptr;
            *result_ptr_i = *(float*)obj_ptr;
        } else {
            float* result_ptr_i = result_ptr;
            *result_ptr_i = (*result_ptr_i > *(float*)obj_ptr) ? *result_ptr_i : *(float*)obj_ptr;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_max(
    tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
    const dynarray* strides_res, const dynarray* transposed_obj_strides,
    const dynarray* res_shape, const dynarray* transposed_obj_shape,
    const size_t dim_size,
    container_type_t type, pml_err_t* err
) {
    do {
        void* result_ptr = iterator_res->get_next(iterator_res, res_shape, strides_res, err);
        for (size_t i = 0; i < dim_size && result_ptr; i++) {
            void* obj_ptr = iterator_obj->get_next(iterator_obj, transposed_obj_shape, transposed_obj_strides, err);
            if (obj_ptr && *err == PML_OK) {
                axis_operation_max_util(result_ptr, obj_ptr, i, type, err);
                if (*err != PML_OK) {
                    return;
                }
            } else {
                return;
            }
        }
    } while (!iterator_res->finished && !iterator_obj->finished);
}

tensor* tensor_axis_max(tensor* tensor, const size_t axis, pml_err_t* err) {
    return tensor_apply_unary_axis(tensor, axis, err, tensor->type, axis_operation_max);
}

static void axis_operation_min_util(void* result_ptr, void* obj_ptr, const size_t iter, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        if (iter == 0) {
            int32_t* result_ptr_i = result_ptr;
            *result_ptr_i = *(int32_t*)obj_ptr;
        } else {
            int32_t* result_ptr_i = result_ptr;
            *result_ptr_i = (*result_ptr_i < *(int32_t*)obj_ptr) ? *result_ptr_i : *(int32_t*)obj_ptr;
        }
        break;
    case TYPE_FLOAT:
        if (iter == 0) {
            float* result_ptr_i = result_ptr;
            *result_ptr_i = *(float*)obj_ptr;
        } else {
            float* result_ptr_i = result_ptr;
            *result_ptr_i = (*result_ptr_i < *(float*)obj_ptr) ? *result_ptr_i : *(float*)obj_ptr;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_min(
    tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
    const dynarray* strides_res, const dynarray* transposed_obj_strides,
    const dynarray* res_shape, const dynarray* transposed_obj_shape,
    const size_t dim_size,
    container_type_t type, pml_err_t* err
) {
    do {
        void* result_ptr = iterator_res->get_next(iterator_res, res_shape, strides_res, err);
        for (size_t i = 0; i < dim_size && result_ptr; i++) {
            void* obj_ptr = iterator_obj->get_next(iterator_obj, transposed_obj_shape, transposed_obj_strides, err);
            if (obj_ptr && *err == PML_OK) {
                axis_operation_min_util(result_ptr, obj_ptr, i, type, err);
                if (*err != PML_OK) {
                    return;
                }
            } else {
                return;
            }
        }
    } while (!iterator_res->finished && !iterator_obj->finished);
}

tensor* tensor_axis_min(tensor* tensor, const size_t axis, pml_err_t* err) {
    return tensor_apply_unary_axis(tensor, axis, err, tensor->type, axis_operation_min);
}

static void axis_operation_mean_util(void* result_ptr, size_t n_elems, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i /= (int32_t)n_elems;
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f /= (float)n_elems;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_mean(
    tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
    const dynarray* strides_res, const dynarray* transposed_obj_strides,
    const dynarray* res_shape, const dynarray* transposed_obj_shape,
    const size_t dim_size,
    container_type_t type, pml_err_t* err
) {
    do {
        void* result_ptr = iterator_res->get_next(iterator_res, res_shape, strides_res, err);
        for (size_t i = 0; i < dim_size && result_ptr; i++) {
            void* obj_ptr = iterator_obj->get_next(iterator_obj, transposed_obj_shape, transposed_obj_strides, err);
            if (obj_ptr && *err == PML_OK) {
                axis_operation_sum_util(result_ptr, obj_ptr, type, err);
                if (*err != PML_OK) {
                    return;
                }
            } else {
                return;
            }
        }
        if (result_ptr) axis_operation_mean_util(result_ptr, dim_size, type, err);
        if (*err != PML_OK) {
            return;
        }
    } while (!iterator_res->finished && !iterator_obj->finished);
}

tensor* tensor_axis_mean(tensor* tensor, const size_t axis, pml_err_t* err) {
    return tensor_apply_unary_axis(tensor, axis, err, tensor->type, axis_operation_mean);
}

static void axis_operation_var_sum_util(void* result_ptr, void** sum_squares, void* obj_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        if (!*sum_squares) {
            *sum_squares = malloc(sizeof(int32_t));
            **(int32_t**)sum_squares = 0;
        }
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i += *(int32_t*)obj_ptr;
        **(int32_t**)sum_squares += *(int32_t*)obj_ptr * *(int32_t*)obj_ptr;
        break;
    case TYPE_FLOAT:
        if (!*sum_squares) {
            *sum_squares = malloc(sizeof(float));
            **(float**)sum_squares = 0;
        }
        float* result_ptr_f = result_ptr;
        *result_ptr_f += *(float*)obj_ptr;
        **(float**)sum_squares += *(float*)obj_ptr * *(float*)obj_ptr;
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_var_util(void* result_ptr, void* sum_squares, size_t n_elem, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_INT32:
        int32_t* result_ptr_i = result_ptr;
        *result_ptr_i = *(int32_t*)sum_squares / (int32_t)n_elem - (*result_ptr_i / (int32_t)n_elem) * (*result_ptr_i / (int32_t)n_elem);
        break;
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = *(float*)sum_squares / (float)n_elem - (*result_ptr_f / (float)n_elem) * (*result_ptr_f / (float)n_elem);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

static void axis_operation_var(
    tensor_iterator* iterator_res, tensor_iterator* iterator_obj,
    const dynarray* strides_res, const dynarray* transposed_obj_strides,
    const dynarray* res_shape, const dynarray* transposed_obj_shape,
    const size_t dim_size,
    container_type_t type, pml_err_t* err
) {
    do {
        void* result_ptr = iterator_res->get_next(iterator_res, res_shape, strides_res, err);
        if (result_ptr) {
            void* sum_squares = NULL;
            for (size_t i = 0; i < dim_size; i++) {
                void* obj_ptr = iterator_obj->get_next(iterator_obj, transposed_obj_shape, transposed_obj_strides, err);
                if (obj_ptr && *err == PML_OK) {
                    axis_operation_var_sum_util(result_ptr, &sum_squares, obj_ptr, type, err);
                    if (*err != PML_OK) {
                        return;
                    }
                } else {
                    return;
                }
            }
            axis_operation_var_util(result_ptr, sum_squares, dim_size, type, err);
            free(sum_squares);
            if (*err != PML_OK) {
                return;
            }
        }
    } while (!iterator_res->finished && !iterator_obj->finished);
}

tensor* tensor_axis_var(tensor* tensor, const size_t axis, pml_err_t* err) {
    return tensor_apply_unary_axis(tensor, axis, err, tensor->type, axis_operation_var);
}

static void tensor_matmul_2d_sum_helper(
    void* result_element_ptr, void* left_row_ptr, void* right_col_ptr,
    int32_t stride_col_left, int32_t stride_row_right,
    int32_t shape_col_left,
    size_t element_size, container_type_t type
) {
    for (size_t i = 0; i < (size_t)shape_col_left; i++) {
        if (i == 0) {
            switch (type)
            {
            case TYPE_INT32:
                *(int32_t*)result_element_ptr = *(int32_t*)left_row_ptr * *(int32_t*)right_col_ptr;
                break;
            case TYPE_FLOAT:
                *(float*)result_element_ptr = *(float*)left_row_ptr * *(float*)right_col_ptr;
                break;
            default:
                return;
                break;
            }
        } else {
            switch (type)
            {
            case TYPE_INT32:
                *(int32_t*)result_element_ptr += *(int32_t*)((char*)left_row_ptr + element_size * i * stride_col_left) \
                    * *(int32_t*)((char*)right_col_ptr + element_size * i * stride_row_right);
                break;
            case TYPE_FLOAT:
                *(float*)result_element_ptr += *(float*)((char*)left_row_ptr + element_size * i * stride_col_left) \
                    * *(float*)((char*)right_col_ptr + element_size * i * stride_row_right);
                break;
            default:
                return;
                break;
            }
        }
    }
}

// Use pointers to left upper corner of matrices
static void tensor_matmul_2d(
    void* result_ptr, void* left_ptr, void* right_ptr,
    int32_t stride_row_result, int32_t stride_col_result,
    int32_t stride_row_left, int32_t stride_col_left,
    int32_t stride_row_right, int32_t stride_col_right,
    int32_t shape_row_result, int32_t shape_col_result,
    int32_t shape_row_left, int32_t shape_col_left,
    int32_t shape_row_right, int32_t shape_col_right,
    container_type_t type, pml_err_t* err
) {
    #ifdef USE_BLAS
    if (type == TYPE_FLOAT) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            shape_row_result, shape_col_result, shape_col_left,
            1.0f,
            (const float*)left_ptr, stride_row_left,
            (const float*)right_ptr, stride_row_right,
            0.0f,
            (float*)result_ptr, stride_row_result);
        *err = PML_OK;
        return;
    }
    #endif

    size_t element_size = 0;
    switch (type)
    {
    case TYPE_INT32:
        element_size = sizeof(int32_t);
        break;
    case TYPE_FLOAT:
        element_size = sizeof(float);
        break;
    default:
        *err = PML_WRONG_TYPE;
        return;
        break;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)shape_row_left; i++) {
        #pragma omp parallel for
        for (size_t j = 0; j < (size_t)shape_col_right; j++) {
            tensor_matmul_2d_sum_helper(
                (char*)result_ptr + i * element_size * stride_row_result + j * element_size * stride_col_result,
                (char*)left_ptr + i * stride_row_left * element_size,
                (char*)right_ptr + j * element_size * stride_col_right,
                stride_col_left, stride_row_right,
                shape_col_left, element_size, type
            );
        }
    }
}

static tensor* tensor_apply_matmul(
    const tensor* left, const tensor* right, const container_type_t type, pml_err_t* err
) {
    if (left->shape._size < 1 || right->shape._size < 1) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    tensor* result = NULL;
    if (left->shape._size <= 2 && right->shape._size <= 2) {
        int reshape_counter = 0;
        bool right_reshaped = false;

        int32_t shape_row_left;
        int32_t shape_col_left;
        int32_t stride_row_left;
        int32_t stride_col_left;

        if (left->shape._size == 1) {
            shape_row_left = 1;
            shape_col_left = *(int32_t*)left->shape._container;
            stride_row_left = 0;
            stride_col_left = *(int32_t*)left->strides._container;
            reshape_counter++;
        } else {
            shape_row_left = *(int32_t*)left->shape._container;
            shape_col_left = *(int32_t*)((char*)left->shape._container + sizeof(int32_t));
            stride_row_left = *(int32_t*)left->strides._container;
            stride_col_left = *(int32_t*)((char*)left->strides._container + sizeof(int32_t));
        }

        int32_t shape_row_right;
        int32_t shape_col_right;
        int32_t stride_row_right;
        int32_t stride_col_right;

        if (right->shape._size == 1) {
            shape_row_right = *(int32_t*)right->shape._container;
            shape_col_right = 1;
            stride_row_right = *(int32_t*)right->strides._container;
            stride_col_right = 0;
            reshape_counter++;
            right_reshaped = true;
        } else {
            shape_row_right = *(int32_t*)right->shape._container;
            shape_col_right = *(int32_t*)((char*)right->shape._container + sizeof(int32_t));
            stride_row_right = *(int32_t*)right->strides._container;
            stride_col_right = *(int32_t*)((char*)right->strides._container + sizeof(int32_t));
        }

        if (shape_col_left != shape_row_right) {
            *err = PML_INCORRECT_INPUT;
            return NULL;
        }

        int32_t raw_result_shape[] = {shape_row_left, shape_col_right};
        dynarray result_shape = dynarray_create(raw_result_shape, 2, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        result = tensor_create_zeros(type, 2, result_shape, err);
        if (*err != PML_OK) {
            dynarray_free(&result_shape);
            return NULL;
        }

        int32_t stride_row_result = *(int32_t*)result->strides._container;
        int32_t stride_col_result = *(int32_t*)((char*)result->strides._container + sizeof(int32_t));

        tensor_matmul_2d(
            result->data, left->data, right->data,
            stride_row_result, stride_col_result,
            stride_row_left, stride_col_left,
            stride_row_right, stride_col_right,
            raw_result_shape[0], raw_result_shape[1],
            shape_row_left, shape_col_left,
            shape_row_right, shape_col_right,
            type, err
        );
        if (*err != PML_OK) {
            tensor_free(result);
            free(result);
            return NULL;
        }

        if (reshape_counter == 2) {
            result->n_dim = 0;
            result->shape.resize(&result->shape, 0);
            result->strides.resize(&result->strides, 0);
        } else if (reshape_counter == 1) {
            result->n_dim = 1;
            if (!right_reshaped) {
                *(int32_t*)result->shape._container = result->shape.get_at(&result->shape, result->shape._size - 1).val.i;
                *(int32_t*)result->strides._container = result->strides.get_at(&result->strides, result->strides._size - 1).val.i;
            }
            result->shape.resize(&result->shape, 1);
            result->strides.resize(&result->strides, 1);
        }

        return result;
    }
    
    int reshape_counter = 0;
    bool right_reshaped = false;

    int32_t shape_row_left;
    int32_t shape_col_left;
    int32_t stride_row_left;
    int32_t stride_col_left;

    if (left->shape._size == 1) {
        shape_row_left = 1;
        shape_col_left = *(int32_t*)left->shape._container;
        stride_row_left = 0;
        stride_col_left = *(int32_t*)left->strides._container;
        reshape_counter++;
    } else {
        shape_row_left = left->shape.get_at(&left->shape, left->shape._size - 2).val.i;
        shape_col_left = left->shape.get_at(&left->shape, left->shape._size - 1).val.i;
        stride_row_left = left->strides.get_at(&left->strides, left->strides._size - 2).val.i;
        stride_col_left = left->strides.get_at(&left->strides, left->strides._size - 1).val.i;
    }

    int32_t shape_row_right;
    int32_t shape_col_right;
    int32_t stride_row_right;
    int32_t stride_col_right;

    if (right->shape._size == 1) {
        shape_row_right = *(int32_t*)right->shape._container;
        shape_col_right = 1;
        stride_row_right = *(int32_t*)right->strides._container;
        stride_col_right = 0;
        reshape_counter++;
        right_reshaped = true;
    } else {
        shape_row_right = right->shape.get_at(&right->shape, right->shape._size - 2).val.i;
        shape_col_right = right->shape.get_at(&right->shape, right->shape._size - 1).val.i;
        stride_row_right = right->strides.get_at(&right->strides, right->strides._size - 2).val.i;
        stride_col_right = right->strides.get_at(&right->strides, right->strides._size - 1).val.i;
    }

    if (shape_col_left != shape_row_right) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }

    int32_t batch_num_shape = (right->shape._size > left->shape._size) ? (right->shape._size - 2) : (left->shape._size - 2);

    dynarray left_batch_shape = dynarray_zeros(
        (size_t)(((int32_t)left->shape._size - 2 > 0) ? (left->shape._size - 2) : 0),
        TYPE_INT32, err
    );
    if (*err != PML_OK) {
        return NULL;
    }
    dynarray left_batch_strides = dynarray_zeros(
        (size_t)(((int32_t)left->shape._size - 2 > 0) ? (left->shape._size - 2) : 0), 
        TYPE_INT32, err
    );
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        return NULL;
    }
    dynarray right_batch_shape = dynarray_zeros(
        (size_t)(((int32_t)right->shape._size - 2 > 0) ? (right->shape._size - 2) : 0), 
        TYPE_INT32, err
    );
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        return NULL;
    }
    dynarray right_batch_strides = dynarray_zeros(
        (size_t)(((int32_t)right->shape._size - 2 > 0) ? (right->shape._size - 2) : 0), 
        TYPE_INT32, err
    );
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        return NULL;
    }

    for (int i = 2; i < batch_num_shape + 2; i++) {
        if (i < (int)left->shape._size) {
            int32_t shape_val = left->shape.get_at(&left->shape, left->shape._size - i - 1).val.i;
            int32_t stride_val = left->strides.get_at(&left->strides, left->strides._size - i - 1).val.i;
            left_batch_shape.set_at(
                &left_batch_shape,
                (size_t)batch_num_shape - 1 - (i - 2),
                &shape_val
            );
            left_batch_strides.set_at(
                &left_batch_strides,
                (size_t)batch_num_shape - 1 - (i - 2),
                &stride_val
            );
        }
        if (i < (int)right->shape._size) {
            int32_t shape_val = right->shape.get_at(&right->shape, right->shape._size - i - 1).val.i;
            int32_t stride_val = right->strides.get_at(&right->strides, right->strides._size - i - 1).val.i;
            right_batch_shape.set_at(
                &right_batch_shape,
                (size_t)batch_num_shape - 1 - (i - 2),
                &shape_val
            );
            right_batch_strides.set_at(
                &right_batch_strides,
                (size_t)batch_num_shape - 1 - (i - 2),
                &stride_val
            );
        }
    }
    
    // Create tensors for compatibility
    tensor left_batch_part = {
        .shape = left_batch_shape,
        .strides = left_batch_strides,
        .n_dim = left_batch_shape._size,
    };
    tensor right_batch_part = {
        .shape = right_batch_shape,
        .strides = right_batch_strides,
        .n_dim = right_batch_shape._size,
    };

    if (!tensor_shapes_broadcastable(&left_batch_part, &right_batch_part, err)) {
        if (*err != PML_OK) {
            dynarray_free(&left_batch_shape);
            dynarray_free(&left_batch_strides);
            dynarray_free(&right_batch_shape);
            dynarray_free(&right_batch_strides);
            return NULL;
        }
        *err = PML_TENSORS_NOT_BROADCASTABLE;
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        return NULL;
    }

    tensor_broadcast_tensors_tuple_t out = tensor_broadcast_tensors(&left_batch_part, &right_batch_part, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        return NULL;
    }

    dynarray result_shape = dynarray_zeros((size_t)batch_num_shape + 2, TYPE_INT32, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        dynarray_free(&out.left_strides);
        dynarray_free(&out.result_shape);
        dynarray_free(&out.right_strides);
        return NULL;
    }

    for (int i = 0; i < (int)result_shape._size; i++) {
        if (i < batch_num_shape) {
            int32_t shape_val = out.result_shape.get_at(&out.result_shape, (size_t)i).val.i;
            result_shape.set_at(&result_shape, i, &shape_val);
        } else if (i == (int)result_shape._size - 2) {
            result_shape.set_at(&result_shape, i, &shape_row_left);
        } else {
            result_shape.set_at(&result_shape, i, &shape_col_right);
        }
    }

    result = tensor_create_zeros(type, result_shape._size, result_shape, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        dynarray_free(&out.left_strides);
        dynarray_free(&out.result_shape);
        dynarray_free(&out.right_strides);
        dynarray_free(&result_shape);
        return NULL;
    }

    tensor_iterator* iterator_left = tensor_iterator_create(left, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        dynarray_free(&out.left_strides);
        dynarray_free(&out.result_shape);
        dynarray_free(&out.right_strides);
        dynarray_free(&result_shape);
        free(iterator_left);
        free(result);
        return NULL;
    }
    tensor_iterator* iterator_right = tensor_iterator_create(right, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        dynarray_free(&out.left_strides);
        dynarray_free(&out.result_shape);
        dynarray_free(&out.right_strides);
        dynarray_free(&result_shape);
        free(iterator_left);
        free(iterator_right);
        free(result);
        return NULL;
    }
    tensor_iterator* iterator_result = tensor_iterator_create(result, err);
    if (*err != PML_OK) {
        dynarray_free(&left_batch_shape);
        dynarray_free(&left_batch_strides);
        dynarray_free(&right_batch_shape);
        dynarray_free(&right_batch_strides);
        dynarray_free(&out.left_strides);
        dynarray_free(&out.result_shape);
        dynarray_free(&out.right_strides);
        dynarray_free(&result_shape);
        free(iterator_left);
        free(iterator_right);
        free(iterator_result);
        free(result);
        return NULL;
    }

    int32_t stride_row_result = result->strides.get_at(&result->strides, result->strides._size - 2).val.i;
    int32_t stride_col_result = result->strides.get_at(&result->strides, result->strides._size - 1).val.i;

    do {
        void* left_ptr = iterator_left->get_next(iterator_left, &out.result_shape, &out.left_strides, err);
        void* right_ptr = iterator_right->get_next(iterator_right, &out.result_shape, &out.right_strides, err);
        void* result_ptr = iterator_result->get_next(iterator_result, &out.result_shape, &result->strides, err);
        if (left_ptr && right_ptr && result_ptr) {
            // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
            tensor_matmul_2d(
                result_ptr, left_ptr, right_ptr,
                stride_row_result, stride_col_result,
                stride_row_left, stride_col_left,
                stride_row_right, stride_col_right,
                shape_row_left, shape_col_right,
                shape_row_left, shape_col_left,
                shape_row_right, shape_col_right,
                type, err
            );
            if (*err != PML_OK) {
                dynarray_free(&left_batch_shape);
                dynarray_free(&left_batch_strides);
                dynarray_free(&right_batch_shape);
                dynarray_free(&right_batch_strides);
                dynarray_free(&out.left_strides);
                dynarray_free(&out.result_shape);
                dynarray_free(&out.right_strides);
                dynarray_free(&result_shape);
                free(iterator_left);
                free(iterator_right);
                free(iterator_result);
                free(result);
                break;
            }
        }
    } while (!iterator_left->finished && !iterator_right->finished && !iterator_result->finished);

    if (reshape_counter != 0) {
        result->n_dim = result->n_dim - 1;
        if (right_reshaped) {
            result->shape.resize(&result->shape, result->shape._size - 1);
            result->strides.resize(&result->strides, result->strides._size - 1);
        } else {
            int32_t shape_val = result->shape.get_at(&result->shape, result->shape._size - 1).val.i;
            int32_t stride_val = result->strides.get_at(&result->strides, result->strides._size - 1).val.i;
            result->shape.resize(&result->shape, result->shape._size - 1);
            result->strides.resize(&result->strides, result->strides._size - 1);
            result->shape.set_at(&result->shape, result->shape._size - 1, &shape_val);
            result->strides.set_at(&result->strides, result->strides._size - 1, &stride_val);
        }
    }

    dynarray_free(&left_batch_shape);
    dynarray_free(&left_batch_strides);
    dynarray_free(&right_batch_shape);
    dynarray_free(&right_batch_strides);
    dynarray_free(&out.left_strides);
    dynarray_free(&out.result_shape);
    dynarray_free(&out.right_strides);
    free(iterator_left);
    free(iterator_right);
    free(iterator_result);
    
    return result;
}

tensor* tensor_matmul(const tensor* left, const tensor* right, pml_err_t* err) {
    if (left->type != right->type) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    return tensor_apply_matmul(left, right, left->type, err);
}

tensor* tensor_sum(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_INT32:
        int32_t res_i = 0;
        int32_t* data_i = (int32_t*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_i += data_i[i];
        }
        result = tensor_create_scalar(&res_i, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    case TYPE_FLOAT:
        float res_f = 0;
        float* data_f = (float*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_f += data_f[i];
        }
        result = tensor_create_scalar(&res_f, TYPE_FLOAT, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

tensor* tensor_max(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_INT32:
        int32_t* data_i = (int32_t*)tens->data;
        int32_t res_i = data_i[0];
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_i = (data_i[i] > res_i) ? data_i[i] : res_i;
        }
        result = tensor_create_scalar(&res_i, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    case TYPE_FLOAT:
        float* data_f = (float*)tens->data;
        float res_f = data_f[0];
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_f = (data_f[i] > res_f) ? data_f[i] : res_f;
        }
        result = tensor_create_scalar(&res_f, TYPE_FLOAT, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

tensor* tensor_min(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_INT32:
        int32_t* data_i = (int32_t*)tens->data;
        int32_t res_i = data_i[0];
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_i = (data_i[i] < res_i) ? data_i[i] : res_i;
        }
        result = tensor_create_scalar(&res_i, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    case TYPE_FLOAT:
        float* data_f = (float*)tens->data;
        float res_f = data_f[0];
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_f = (data_f[i] < res_f) ? data_f[i] : res_f;
        }
        result = tensor_create_scalar(&res_f, TYPE_FLOAT, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

tensor* tensor_mean(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_INT32:
        int32_t res_i = 0;
        int32_t* data_i = (int32_t*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_i += data_i[i];
        }
        res_i /= (int32_t)tens->data_num_elems;
        result = tensor_create_scalar(&res_i, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    case TYPE_FLOAT:
        float res_f = 0;
        float* data_f = (float*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_f += data_f[i];
        }
        res_f /= (float)tens->data_num_elems;
        result = tensor_create_scalar(&res_f, TYPE_FLOAT, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

tensor* tensor_var(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_INT32:
        int32_t res_i = 0;
        int32_t res_i_2 = 0;
        int32_t* data_i = (int32_t*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_i += data_i[i];
            res_i_2 += data_i[i] * data_i[i];
        }
        res_i /= (int32_t)tens->data_num_elems;
        res_i_2 /= (int32_t)tens->data_num_elems;
        int32_t res_var_i = res_i_2 - res_i * res_i;
        result = tensor_create_scalar(&res_var_i, TYPE_INT32, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    case TYPE_FLOAT:
        float res_f = 0;
        float res_f_2 = 0;
        float* data_f = (float*)tens->data;
        for (size_t i = 0; i < tens->data_num_elems; i++) {
            res_f += data_f[i];
            res_f_2 += data_f[i] * data_f[i];
        }
        res_f /= (float)tens->data_num_elems;
        res_f_2 /= (float)tens->data_num_elems;
        float res_var_f = res_f_2 - res_f * res_f;
        result = tensor_create_scalar(&res_var_f, TYPE_FLOAT, err);
        if (*err != PML_OK) {
            return NULL;
        }
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

static tensor* tensor_apply_elementwise_unary_operation(
    const tensor* input, pml_err_t* err, container_type_t type,
    void (*unary_operation)(void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err)
) {
    *err = PML_OK;
    tensor* result;

    dynarray result_shape = dynarray_clone(&input->shape, err);
    if (*err != PML_OK) {
        return NULL;
    }
    result = tensor_create_zeros(type, input->shape._size, result_shape, err);
    if (*err != PML_OK) {
        dynarray_free(&result_shape);
        return NULL;
    }
    tensor_iterator* iterator_input = tensor_iterator_create(input, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        return NULL;
    }
    tensor_iterator* iterator_result = tensor_iterator_create(result, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        free(iterator_input);
        return NULL;
    }
    do {
        void* input_ptr = iterator_input->get_next(iterator_input, &input->shape, &input->strides, err);
        void* result_ptr = iterator_result->get_next(iterator_result, &result->shape, &result->strides, err);
        if (input_ptr && result_ptr) {
            unary_operation(result_ptr, input_ptr, type, err);
            if (*err != PML_OK) {
                tensor_free(result);
                free(result);
                break;
            }
            // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
        }
    } while (!iterator_input->finished && !iterator_result->finished);
    tensor_iterator_free(iterator_input);
    free(iterator_input);
    tensor_iterator_free(iterator_result);
    free(iterator_result);

    return result;
}

static void tensor_sqrt_unary_operation(void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = sqrtf(*(float*)input_ptr);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_sqrt(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_FLOAT:
        result = tensor_apply_elementwise_unary_operation(tens, err, TYPE_FLOAT, tensor_sqrt_unary_operation);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

static void tensor_log_unary_operation(void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = logf(*(float*)input_ptr);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_log(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_FLOAT:
        result = tensor_apply_elementwise_unary_operation(tens, err, TYPE_FLOAT, tensor_log_unary_operation);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

static void tensor_exp_unary_operation(void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = expf(*(float*)input_ptr);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_exp(const tensor* tens, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_FLOAT:
        result = tensor_apply_elementwise_unary_operation(tens, err, TYPE_FLOAT, tensor_exp_unary_operation);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

static tensor* tensor_apply_elementwise_unary_pow(
    const tensor* input, float exponent, pml_err_t* err, container_type_t type,
    void (*unary_pow)(void* result_ptr, void* input_ptr, float exponent, container_type_t type, pml_err_t* err)
) {
    *err = PML_OK;
    tensor* result;

    dynarray result_shape = dynarray_clone(&input->shape, err);
    if (*err != PML_OK) {
        return NULL;
    }
    result = tensor_create_zeros(type, input->shape._size, result_shape, err);
    if (*err != PML_OK) {
        dynarray_free(&result_shape);
        return NULL;
    }
    tensor_iterator* iterator_input = tensor_iterator_create(input, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        return NULL;
    }
    tensor_iterator* iterator_result = tensor_iterator_create(result, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        free(iterator_input);
        return NULL;
    }
    do {
        void* input_ptr = iterator_input->get_next(iterator_input, &input->shape, &input->strides, err);
        void* result_ptr = iterator_result->get_next(iterator_result, &result->shape, &result->strides, err);
        if (input_ptr && result_ptr) {
            unary_pow(result_ptr, input_ptr, exponent, type, err);
            if (*err != PML_OK) {
                tensor_free(result);
                free(result);
                break;
            }
            // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
        }
    } while (!iterator_input->finished && !iterator_result->finished);
    tensor_iterator_free(iterator_input);
    free(iterator_input);
    tensor_iterator_free(iterator_result);
    free(iterator_result);

    return result;
}

static void tensor_pow_unary_operation(void* result_ptr, void* input_ptr, float exponent, container_type_t type, pml_err_t* err) {
    switch (type)
    {
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = powf(*(float*)input_ptr, exponent);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_pow(const tensor* tens, float exponent, pml_err_t* err) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_FLOAT:
        result = tensor_apply_elementwise_unary_pow(tens, exponent, err, TYPE_FLOAT, tensor_pow_unary_operation);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}

static tensor* tensor_apply_elementwise_unary_custom_float_operation(
    const tensor* input, pml_err_t* err, container_type_t type,
    void (*unary_operation)(void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err, float (*custom_float_operation)(float __x)),
    float (*custom_float_operation)(float __x)
) {
    *err = PML_OK;
    tensor* result;

    dynarray result_shape = dynarray_clone(&input->shape, err);
    if (*err != PML_OK) {
        return NULL;
    }
    result = tensor_create_zeros(type, input->shape._size, result_shape, err);
    if (*err != PML_OK) {
        dynarray_free(&result_shape);
        return NULL;
    }
    tensor_iterator* iterator_input = tensor_iterator_create(input, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        return NULL;
    }
    tensor_iterator* iterator_result = tensor_iterator_create(result, err);
    if (*err != PML_OK) {
        tensor_free(result);
        free(result);
        free(iterator_input);
        return NULL;
    }
    do {
        void* input_ptr = iterator_input->get_next(iterator_input, &input->shape, &input->strides, err);
        void* result_ptr = iterator_result->get_next(iterator_result, &result->shape, &result->strides, err);
        if (input_ptr && result_ptr) {
            unary_operation(result_ptr, input_ptr, type, err, custom_float_operation);
            if (*err != PML_OK) {
                tensor_free(result);
                free(result);
                break;
            }
            // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
        }
    } while (!iterator_input->finished && !iterator_result->finished);
    tensor_iterator_free(iterator_input);
    free(iterator_input);
    tensor_iterator_free(iterator_result);
    free(iterator_result);

    return result;
}

static void tensor_custom_unary_operation(
    void* result_ptr, void* input_ptr, container_type_t type, pml_err_t* err,
    float (*custom_float_operation)(float __x)
) {
    switch (type)
    {
    case TYPE_FLOAT:
        float* result_ptr_f = result_ptr;
        *result_ptr_f = custom_float_operation(*(float*)input_ptr);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
}

tensor* tensor_float_custom_elementwise_unary_operation(
    const tensor* tens, pml_err_t* err, 
    float (*operation)(float __x)
) {
    tensor* result;
    switch (tens->type)
    {
    case TYPE_FLOAT:
        result = tensor_apply_elementwise_unary_custom_float_operation(tens, err, TYPE_FLOAT, tensor_custom_unary_operation, operation);
        break;
    default:
        *err = PML_WRONG_TYPE;
        break;
    }
    return result;
}