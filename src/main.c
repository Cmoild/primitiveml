#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>


float custom(float x){
    return x * 23.64;
}


int main(){
    pml_err_t err = PML_OK;

    dynarray shape_test = dynarray_create((int32_t[]){2, 1, 3}, 3, TYPE_INT32, &err);
    tensor* test = tensor_create((float[]){1., 2., 3., 4., 5., 6.}, 6, TYPE_FLOAT, 3, shape_test, &err);

    tensor* s = softmax(test, -1, &err);
    s->print(s);


    return 0;
}
