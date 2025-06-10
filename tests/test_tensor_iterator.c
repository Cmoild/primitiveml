#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    float raw_data = 1235.235f;
    pml_err_t err = PML_OK;
    tensor* tensor = tensor_create_scalar(&raw_data, TYPE_FLOAT, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tensor->print(tensor);
    tensor_iterator* iterator = tensor_iterator_create(tensor, &err);
    if (err != PML_OK) {
        printf("Error iter create: %d\n", err);
        return 1;
    }
    int32_t new_shape[] = {2, 2};
    dynarray shape = dynarray_create(new_shape, 2, TYPE_INT32, &err);
    int32_t new_strides[] = {0, 0};
    dynarray strides = dynarray_create(new_strides, 2, TYPE_INT32, &err);
    while (!iterator->finished) {
        void* ptr = iterator->get_next(iterator, &shape, &strides, &err);
        if (err != PML_OK) {
            printf("Error iter: %d\n", err);
            return 1;
        }
        if (ptr)
        printf("Cur elem: %f\n", *(float*)ptr);
        iterator->current_indices.print(&iterator->current_indices);
    }
    tensor_free(tensor);
    dynarray_free(&shape);
    dynarray_free(&strides);
    free(tensor);
    return 0;
}
