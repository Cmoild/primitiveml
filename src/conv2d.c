#include <conv2d.h>
#include <module.h>
#include <stdlib.h>
#include <stdio.h>


static void conv2d_module_free(void* self);

static void conv2d_module_print(const void* self);

static tensor* conv2d_module_forward(const void* self, const tensor* input);

conv2d_module* conv2d_module_create(tensor* weight, tensor* bias, int32_t padding, int32_t stride, pml_err_t* err) {
    
}