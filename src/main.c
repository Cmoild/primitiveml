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
    // int32_t shape_bias_raw[] = {5};
    // dynarray shape_bias = dynarray_create(shape_bias_raw, 1, TYPE_INT32, &err);
    // float bias_raw[5] = {-3.7451, -4.8251, -5.6107, -1.8519, -2.2098};
    // tensor* bias = tensor_create(bias_raw, 5, TYPE_FLOAT, 1, shape_bias, &err);

    dynarray shape_test = dynarray_create((int32_t[]){2, 3}, 2, TYPE_INT32, &err);
    tensor* test = tensor_create((float[]){1., 2., 3., 4., 5., 6.}, 6, TYPE_FLOAT, 2, shape_test, &err);

    tensor* s = sigmoid(test, &err);
    s->print(s);


    // int32_t shape_weights_raw[] = {5, 1};
    // dynarray shape_weights = dynarray_create(shape_weights_raw, 2, TYPE_INT32, &err);
    // float weights_raw[] = {-1.6920, 1.6244, -1.5764, -2.3538, 2.2618};
    // tensor* weights = tensor_create(weights_raw, 5, TYPE_FLOAT, 2, shape_weights, &err);

    // dynarray shape_input = dynarray_create((int32_t[]){1}, 1, TYPE_INT32, &err);
    // tensor* input = tensor_create((float[]){5.}, 1, TYPE_FLOAT, 1, shape_input, &err);

    // linear_module* fc = linear_module_create(weights, bias, &err);

    // tensor* output1 = fc->module_base.forward(fc, input);

    // tensor* output2 = relu(output1, &err);

    // dynarray shape_bias2 = dynarray_create((int32_t[]){1}, 1, TYPE_INT32, &err);
    // tensor* bias2 = tensor_create((float[]){0.2878}, 1, TYPE_FLOAT, 1, shape_bias2, &err);

    // dynarray shape_weights2 = dynarray_create((int32_t[]){1, 5}, 2, TYPE_INT32, &err);
    // tensor* weights2 = tensor_create((float[]){1.6556, 2.4228, 1.7815, 1.2710, 1.7468}, 5, TYPE_FLOAT, 2, shape_weights2, &err);

    // linear_module* fc2 = linear_module_create(weights2, bias2, &err);

    // tensor* output3 = fc2->module_base.forward(fc2, output2);

    // output3->print(output3);

    return 0;
}
