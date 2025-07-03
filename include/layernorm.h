#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <module.h>
#include <error_handling.h>

typedef struct layernorm {
    module_iface module_base;
    tensor* weight;
    tensor* bias;
    tensor* eps;
} layernorm;

layernorm* layernorm_create(tensor* weight, tensor* bias, float eps, pml_err_t* err);

#endif // LAYERNORM_H