#include <linear.h>
#include <module.h>
#include <stdlib.h>
#include <stdio.h>


static void linear_module_free(void* self);

static void linear_module_print(const void* self);

static tensor* linear_module_forward(const void* self, const tensor* input);

linear_module* linear_module_create(tensor* weight, tensor* bias, pml_err_t* err) {
    if (weight->n_dim != 2 && bias->n_dim != 1) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    linear_module* fc = (linear_module*)malloc(sizeof(linear_module));
    fc->module_base = (module_iface){
        .destroy = linear_module_free,
        .forward = linear_module_forward,
        .print = linear_module_print,
    };
    fc->weight = weight;
    fc->bias = bias;
    *err = PML_OK;
    return fc;
}

static void linear_module_free(void* self) {
    linear_module* fc = (linear_module*)self;
    tensor_free(fc->weight);
    free(fc->weight);
    tensor_free(fc->bias);
    free(fc->bias);
    free(fc);
}

static tensor* linear_module_forward(const void* self, const tensor* input) {
    pml_err_t err = PML_OK;
    linear_module* module = (linear_module*)self;
    tensor* weights_transposed = module->weight->transpose(module->weight, 0, 1, &err);
    if (err != PML_OK) {
        // maybe call exit?
        return NULL;
    }
    tensor* mmul = tensor_matmul(input, weights_transposed, &err);
    if (err != PML_OK) {
        return NULL;
    }
    tensor_free(weights_transposed);
    free(weights_transposed);
    tensor* output = tensor_add(mmul, module->bias, module->bias->type, &err);
    if (err != PML_OK) {
        return NULL;
    }
    tensor_free(mmul);
    free(mmul);
    return output;
}

static void linear_module_print(const void* self) {
    printf("Linear module\n");
}