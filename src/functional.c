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

tensor* softmax(const tensor* input, int32_t dim, pml_err_t* err) {
    tensor* exp_input = tensor_exp(input, err);
    if (*err != PML_OK) {
        return NULL;
    }
    if (dim < 0) {
        dim = exp_input->n_dim + dim;
    }
    tensor* exp_sum = tensor_axis_sum(exp_input, (size_t)dim, err);
    if (*err != PML_OK) {
        tensor_free(exp_input);
        free(exp_input);
        return NULL;
    }
    // keepdim=True
    tensor* exp_sum_unsq = exp_sum->unsqueeze(exp_sum, dim, err);
    if (*err != PML_OK) {
        tensor_free(exp_input);
        free(exp_input);
        tensor_free(exp_sum);
        free(exp_sum);
        return NULL;
    }
    tensor* result = tensor_divide(exp_input, exp_sum_unsq, input->type, err);
    tensor_free(exp_input);
    free(exp_input);
    tensor_free(exp_sum);
    free(exp_sum);
    tensor_free(exp_sum_unsq);
    free(exp_sum_unsq);

    return result;
}