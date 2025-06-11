#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t raw_shape1[] = {2, 2};
    dynarray shape1 = dynarray_create(raw_shape1, 2, TYPE_INT32, &err);
    int32_t raw_data1[] = {1, 2, 3, 4};
    tensor* tensor1 = tensor_create(raw_data1, 4, TYPE_INT32, 2, shape1, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    int32_t raw_shape2[] = {2};
    dynarray shape2 = dynarray_create(raw_shape2, 1, TYPE_INT32, &err);
    int32_t raw_data2[] = {1, 2};
    tensor* tensor2 = tensor_create(raw_data2, 2, TYPE_INT32, 1, shape2, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tensor1->print(tensor1);
    tensor2->print(tensor2);

    tensor* result = tensor_add(tensor1, tensor2, TYPE_INT32, &err);
    result->print(result);
    
    tensor_free(tensor1);
    free(tensor1);
    tensor_free(tensor2);
    free(tensor2);
    tensor_free(result);
    free(result);
    return 0;
}
