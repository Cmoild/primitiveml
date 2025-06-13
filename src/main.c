#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t shape_init_raw[] = {2,2,7};
    dynarray shape_init = dynarray_create(shape_init_raw, 3, TYPE_INT32, &err);
    float data[] = {1,1,1,2,1,1,1,1,1,1,1,10,1,1,-1,1,1,1,1,1,1,0,1,1,1,1,1,4};
    tensor* W = tensor_create(data, 28, TYPE_FLOAT, 3, shape_init, &err);
    W->print(W);
    tensor* res = tensor_pow(W, 3.1f, &err);
    res->print(res);
    
    tensor_free(W);
    free(W);
    return 0;
}
