#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t shape_init_raw[] = {4};
    dynarray shape_init = dynarray_create(shape_init_raw, 1, TYPE_INT32, &err);
    float data[] = {0.5, -1.2, 3.3, 0.7};
    tensor* W = tensor_create(data, 4, TYPE_FLOAT, 1, shape_init, &err);
    W->print(W);

    float bias = 2.0;
    tensor* b = tensor_create_scalar(&bias, TYPE_FLOAT, &err);

    int32_t input_shape_raw[] = {4};
    dynarray input_shape = dynarray_create(input_shape_raw, 1, TYPE_INT32, &err);

    float input_data[] = {1, 2, 3, 4};
    tensor* input = tensor_create(input_data, 4, TYPE_FLOAT, 1, input_shape, &err);
    input->print(input);

    tensor* dp = tensor_matmul(W, input, &err);
    dp->print(dp);
    tensor* res = tensor_add(dp, b, TYPE_FLOAT, &err);

    res->print(res);

    tensor_free(res);
    free(res);
    tensor_free(W);
    free(W);
    return 0;
}
