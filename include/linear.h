#ifndef LINEAR_MODULE_H
#define LINEAR_MODULE_H

#include <module.h>
#include <error_handling.h>

typedef struct linear_module {
    module_iface module_base;
    tensor* W;
    tensor* b;
} linear_module;

linear_module* linear_module_create(tensor* W, tensor* b);

#endif // LINEAR_MODULE_H