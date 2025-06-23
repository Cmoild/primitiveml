#include <functional.h>
#include <stdlib.h>
#include <math.h>


tensor* relu(const tensor* input, pml_err_t* err) {
    tensor* zero = tensor_create_scalar((float[]){0.}, TYPE_FLOAT, err);
    if (*err != PML_OK) {
        return NULL;
    }
    tensor* output = tensor_max_binary(input, zero, err);
    tensor_free(zero);
    free(zero);
    return output;
}

float scalar_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

tensor* sigmoid(const tensor* input, pml_err_t* err) {
    tensor* result = tensor_float_custom_elementwise_unary_operation(input, err, scalar_sigmoid);
    return result;
}