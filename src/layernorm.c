#include <layernorm.h>
#include <stdlib.h>
#include <stdio.h>


static void layernorm_free(void* self);

static void layernorm_print(const void* self);

static tensor* layernorm_forward(const void* self, const tensor* input);

layernorm* layernorm_create(tensor* weight, tensor* bias, float eps, pml_err_t* err) {
    if (!(weight->n_dim == 1 && bias->n_dim == 1 && eps > 0.0F)) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    tensor* tensor_eps = tensor_create_scalar(&eps, TYPE_FLOAT, err);
    if (*err != PML_OK) {
        return NULL;
    }
    layernorm* module = (layernorm*)malloc(sizeof(layernorm));
    module->module_base = (module_iface){
        .destroy = layernorm_free,
        .forward = layernorm_forward,
        .print = layernorm_print,
    };
    module->weight = weight;
    module->bias = bias;
    module->eps = tensor_eps;
    return module;
}

static void layernorm_free(void* self) {
    layernorm* module = (layernorm*)self;
    tensor_free(module->bias);
    free(module->bias);
    tensor_free(module->eps);
    free(module->eps);
    tensor_free(module->weight);
    free(module->weight);
}

static void layernorm_print(const void* self) {
    printf("Layernorm\n");
}

static tensor* layernorm_forward(const void* self, const tensor* input) {
    layernorm* module = (layernorm*)self;
    pml_err_t err;
    
    tensor* inp_mean = tensor_axis_mean(input, input->n_dim - 1, &err);
    tensor* inp_mean_keepd = inp_mean->unsqueeze(inp_mean, input->n_dim - 1, &err);

    tensor* inp_var = tensor_axis_var(input, input->n_dim - 1, &err);
    tensor* inp_var_keepd = inp_var->unsqueeze(inp_var, input->n_dim - 1, &err);
    tensor* inp_var_keepd_eps = tensor_add(inp_var_keepd, module->eps, inp_var_keepd->type, &err);
    tensor* var_sqrt = tensor_sqrt(inp_var_keepd_eps, &err);

    tensor_free(inp_var_keepd_eps); free(inp_var_keepd_eps);
    tensor_free(inp_var_keepd); free(inp_var_keepd);
    tensor_free(inp_var); free(inp_var);

    tensor* sub = tensor_subtract(input, inp_mean_keepd, input->type, &err);

    tensor_free(inp_mean_keepd); free(inp_mean_keepd);
    tensor_free(inp_mean); free(inp_mean);

    tensor* norm = tensor_divide(sub, var_sqrt, sub->type, &err);
    tensor_free(sub); free(sub);
    tensor_free(var_sqrt); free(var_sqrt);

    tensor* mul = tensor_multiply(norm, module->weight, norm->type, &err);
    tensor_free(norm); free(norm);

    tensor* result = tensor_add(mul, module->bias, mul->type, &err);
    tensor_free(mul); free(mul);

    return result;
}
