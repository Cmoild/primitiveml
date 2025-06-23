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

    dynarray shape_test = dynarray_create((int32_t[]){2, 3}, 2, TYPE_INT32, &err);
    tensor* test = tensor_create((float[]){1., 2., 3., 4., 5., 6.}, 6, TYPE_FLOAT, 2, shape_test, &err);

    test = test->transpose(test, 0, 1, &err);

    tensor* s = test->unsqueeze(test, -1, &err);
    s->print(s);


    return 0;
}
