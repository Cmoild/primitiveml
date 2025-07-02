#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>
#include <tensor_index.h>


int main(){
    pml_err_t err = PML_OK;

    dynarray shape1 = dynarray_create((int32_t[]){1000,1000}, 2, TYPE_INT32, &err);
    tensor* t1 = tensor_create_zeros(TYPE_FLOAT, 2, shape1, &err);

    dynarray shape2 = dynarray_create((int32_t[]){1000,1000}, 2, TYPE_INT32, &err);
    tensor* t2 = tensor_create_zeros(TYPE_FLOAT, 2, shape2, &err);

    tensor* t3 = tensor_multiply(t1, t2, TYPE_FLOAT, &err);
    t3->shape.print(&t3->shape);
    
    return 0;
}
