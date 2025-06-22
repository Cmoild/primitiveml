#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t shape_init_raw[] = {2,3,4};
    dynarray shape_init = dynarray_create(shape_init_raw, 3, TYPE_INT32, &err);
    float data[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f, 0.07f,
        0.08f, 0.09f, 0.11f, 0.12f,
        0.13f, 0.14f, 0.15f, 0.16f,
    };
    tensor* W = tensor_create(data, 24, TYPE_FLOAT, 3, shape_init, &err);
    W->print(W);
    
    int32_t shape_view_raw[] = {2,3,2,2};
    dynarray shape_view = dynarray_create(shape_view_raw, 4, TYPE_INT32, &err);
    tensor* w_heads = W->view(W, shape_view, &err);
    w_heads->print(w_heads);

    w_heads = w_heads->transpose(w_heads, -2, -3, &err);

    w_heads->print(w_heads);

    // float ex_raw[] = {1, 2, 3, 4};
    // int32_t shape_ex_raw[] = {2, 2};
    // dynarray shape_ex = dynarray_create(shape_ex_raw, 2, TYPE_INT32, &err);
    // tensor* ex = tensor_create(ex_raw, 4, TYPE_FLOAT, 2, shape_ex, &err);
    // ex->print(ex);

    // tensor* res = tensor_matmul(w_heads, ex, &err);
    // res->print(res);

    return 0;
}
