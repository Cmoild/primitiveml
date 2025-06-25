#ifndef CONV2D_MODULE_H
#define CONV2D_MODULE_H

#include <module.h>
#include <error_handling.h>

typedef struct conv2d_module {
    module_iface module_base;
    tensor* weight;
    tensor* bias;
    size_t padding;
} conv2d_module;

conv2d_module* conv2d_module_create(tensor* weight, tensor* bias, size_t padding, pml_err_t* err);

#endif // CONV2D_MODULE_H