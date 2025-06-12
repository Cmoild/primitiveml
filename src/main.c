#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t shape_init_raw[] = {1};
    dynarray shape_init = dynarray_create(shape_init_raw, 1, TYPE_INT32, &err);
    int32_t data[] = {3};
    tensor* tens = tensor_create(data, 1, TYPE_INT32, 1, shape_init, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tens->print(tens);

    int32_t shape_init_raw2[] = {1, 1};
    dynarray shape_init2 = dynarray_create(shape_init_raw2, 2, TYPE_INT32, &err);
    int32_t data2[] = {2};
    tensor* tens2 = tensor_create(data2, 1, TYPE_INT32, 2, shape_init2, &err);
    tens2->print(tens2);

    tensor* res = tensor_matmul(tens, tens2, &err);

    res->print(res);

    tensor_free(res);
    free(res);
    tensor_free(tens);
    free(tens);
    tensor_free(tens2);
    free(tens2);
    return 0;
}
