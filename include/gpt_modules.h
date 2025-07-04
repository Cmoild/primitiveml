#ifndef GPT_MODULES_H
#define GPT_MODULES_H

#include <module.h>
#include <linear.h>
#include <layernorm.h>
#include <embedding.h>
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

typedef struct feed_forward {
    module_iface module_base;
    linear_module* fc;
    linear_module* fc_proj;
} feed_forward;

feed_forward* feed_forward_create(linear_module* fc, linear_module* fc_proj, pml_err_t* err);

typedef struct gpt_block {
    module_iface module_base;
    layernorm* ln_1;
    layernorm* ln_2;
    causal_self_attention* attn;
    feed_forward* ffn;
} gpt_block;

gpt_block* gpt_block_create(layernorm* ln_1, layernorm* ln_2, causal_self_attention* attn, feed_forward* ffn);

typedef struct gpt_model {
    module_iface module_base;
    embedding_module* wte;
    embedding_module* wpe;
    gpt_block** blocks;
    size_t num_blocks;
    layernorm* ln;
    linear_module* fc;
} gpt_model;

gpt_model* gpt_model_create(
    embedding_module* wte, embedding_module* wpe,
    gpt_block** blocks, size_t num_blocks,
    layernorm* ln, linear_module* fc
);

#endif // GPT_MODULES_H