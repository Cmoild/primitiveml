#ifndef LINEAR_MODULE_H
#define LINEAR_MODULE_H

#include <module.h>
#include <error_handling.h>

typedef struct linear_module {
    module_iface module_base;
    tensor* weight;
    tensor* bias;
} linear_module;

linear_module* linear_module_create(tensor* weight, tensor* bias, pml_err_t* err);

#endif // LINEAR_MODULE_H