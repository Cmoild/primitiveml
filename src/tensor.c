#include <tensor.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


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
    if (shape._size == 0) {
        *error = PML_INCORRECT_INPUT;
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
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    if (n_dimentions != 0) {
        err_strides = strides.resize(&strides, n_dimentions);
        if (err_strides != PML_OK) {
            *error = err_strides;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = n_dimentions - 1; i >= 0; i--) {
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
    tnsr->data_num_elems = n_elems;
    tnsr->strides = strides;
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
    return tnsr;
}

tensor* tensor_create_zeros(
    const container_type_t type, 
    const size_t n_dimentions, 
    const dynarray shape, 
    pml_err_t* error) {
    
    if (shape._size == 0) {
        *error = PML_INCORRECT_INPUT;
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
        dynarray_free(&strides);
        free(tnsr);
        return NULL;
    }
    if (n_dimentions != 0) {
        err_strides = strides.resize(&strides, n_dimentions);
        if (err_strides != PML_OK) {
            *error = err_strides;
            dynarray_free(&strides);
            free(tnsr);
            return NULL;
        }
    }
    size_t n_elems = 1;
    for (int i = n_dimentions - 1; i >= 0; i--) {
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
    tnsr->data_num_elems = n_elems;
    tnsr->strides = strides;
    return tnsr;
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
        for (size_t i = 0; i < dim_size.val.i; i++) {
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
        for (size_t i = 0; i < dim_size.val.i; i++) {
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
            if (i < dim_size.val.i - 1) {
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
    if (obj->data) {
        free(obj->data);
    }
}

bool tensor_shapes_broadcastable(tensor* left, tensor* right, pml_err_t* err) {
    for (int i = 0; i < left->shape._size && i < right->shape._size; i++) {
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
tensor_broadcast_tensors(const tensor* left, tensor* right, pml_err_t* err) {
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
        if (i < left->shape._size && i < right->shape._size) {
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
        } else if (i < left->shape._size && i >= right->shape._size) {
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
        } else if (i >= left->shape._size && i < right->shape._size) {
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
            free(result);
            return NULL;
        }
        tensor_iterator* iterator_left = tensor_iterator_create(left, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            tensor_iterator_free(iterator_left);
            free(iterator_left);
            return NULL;
        }
        tensor_iterator* iterator_right = tensor_iterator_create(right, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            tensor_iterator_free(iterator_left);
            free(iterator_left);
            tensor_iterator_free(iterator_right);
            free(iterator_right);
            return NULL;
        }
        tensor_iterator* iterator_result = tensor_iterator_create(result, err);
        if (*err != PML_OK) {
            dynarray_free(&out.left_strides);
            dynarray_free(&out.right_strides);
            dynarray_free(&out.result_shape);
            free(result);
            tensor_iterator_free(iterator_left);
            free(iterator_left);
            tensor_iterator_free(iterator_right);
            free(iterator_right);
            tensor_iterator_free(iterator_result);
            free(iterator_result);
            return NULL;
        }
        do {
            void* left_ptr = iterator_left->get_next(iterator_left, &out.result_shape, &out.left_strides, err);
            void* right_ptr = iterator_right->get_next(iterator_right, &out.result_shape, &out.right_strides, err);
            void* result_ptr = iterator_result->get_next(iterator_result, &out.result_shape, &result->strides, err);
            if (left_ptr && right_ptr && result_ptr) {
                operation(result_ptr, left_ptr, right_ptr, type, err);
                if (*err != PML_OK) {
                    break;
                }
                // printf("left: %d, right: %d, result: %d\n", *(int32_t*)left_ptr, *(int32_t*)right_ptr, *(int32_t*)result_ptr);
            }
        } while (!iterator_left->finished && !iterator_right->finished && !iterator_result->finished);
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
