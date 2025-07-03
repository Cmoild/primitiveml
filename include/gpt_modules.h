#ifndef GPT_MODULES_H
#define GPT_MODULES_H

#include <module.h>
#include <linear.h>
#include <error_handling.h>

typedef struct causal_self_attention {
    module_iface module_base;
    linear_module* qkv_proj;
    linear_module* output_proj;
    tensor* causal_mask;
    size_t embedding_dim;
    size_t num_heads;
} causal_self_attention;

causal_self_attention* causal_self_attention_create(
    linear_module* qkv_proj, linear_module* output_proj, tensor* causal_mask,
    size_t num_heads, size_t embedding_dim, pml_err_t* err
);



#endif // GPT_MODULES_H