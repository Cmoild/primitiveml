#ifndef GPT_MODULES_H
#define GPT_MODULES_H

#include <module.h>
#include <linear.h>
#include <error_handling.h>

typedef struct mha_module {
    linear_module* fc_kqv;
    linear_module* fc_out;
    tensor* attention_mask;
} mha_module;

mha_module* mha_module_create(
    linear_module* fc_kqv, linear_module* fc_out, 
    size_t context_window_size, size_t n_head, pml_err_t* err
);

#endif // GPT_MODULES_H