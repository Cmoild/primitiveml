#include <gpt_modules.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <functional.h>


static void causal_self_attention_free(void* self);

static void causal_self_attention_print(const void* self);

static tensor* causal_self_attention_forward(const void* self, const tensor* input);

causal_self_attention* causal_self_attention_create(
    linear_module* qkv_proj, linear_module* output_proj, tensor* causal_mask,
    size_t num_heads, size_t embedding_dim, pml_err_t* err
) {
    int32_t atten_qkv_inp_sz = qkv_proj->weight->shape.get_at(&qkv_proj->weight->shape, 1).val.i;
    int32_t atten_qkv_out_sz = qkv_proj->weight->shape.get_at(&qkv_proj->weight->shape, 0).val.i;
    int32_t output_proj_inp_sz = output_proj->weight->shape.get_at(&output_proj->weight->shape, 1).val.i;
    int32_t output_proj_out_sz = output_proj->weight->shape.get_at(&output_proj->weight->shape, 0).val.i;
    if (!(atten_qkv_inp_sz * 3 == atten_qkv_out_sz \
        && output_proj_inp_sz == output_proj_out_sz \
        && atten_qkv_inp_sz == output_proj_inp_sz \
        && atten_qkv_inp_sz == (int32_t)embedding_dim \
        && embedding_dim % num_heads == 0)) 
    {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    causal_self_attention* module = (causal_self_attention*)malloc(sizeof(causal_self_attention));
    module->module_base = (module_iface){
        .destroy = causal_self_attention_free,
        .forward = causal_self_attention_forward,
        .print = causal_self_attention_print,
    };
    module->causal_mask = causal_mask;
    module->embedding_dim = embedding_dim;
    module->num_heads = num_heads;
    module->output_proj = output_proj;
    module->qkv_proj = qkv_proj;
    *err = PML_OK;
    return module;
}

static void causal_self_attention_free(void* self) {
    causal_self_attention* module = (causal_self_attention*)self;
    tensor_free(module->causal_mask);
    free(module->causal_mask);
    module->output_proj->module_base.destroy(module->output_proj);
    module->qkv_proj->module_base.destroy(module->qkv_proj);
    free(module);
}

static tensor* causal_self_attention_forward(const void* self, const tensor* input) {
    pml_err_t err = PML_OK;
    causal_self_attention* module = (causal_self_attention*)self;
    if (input->n_dim != 3) {
        return NULL;
    }
    int32_t batch_sz, seq_len, embed_dim;
    batch_sz = input->shape.get_at(&input->shape, 0).val.i;
    seq_len = input->shape.get_at(&input->shape, 1).val.i;
    embed_dim = input->shape.get_at(&input->shape, 2).val.i;
    if (embed_dim != (int32_t)module->embedding_dim) {
        return NULL;
    }

    tensor* qkv = module->qkv_proj->module_base.forward(module->qkv_proj, input);
    index_tuple_t tup_q = {
        .len = 3,
        .items = (tensor_index_t[]) {
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = batch_sz } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = seq_len } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = embed_dim } } },
        },
    };
    index_tuple_t tup_k = {
        .len = 3,
        .items = (tensor_index_t[]) {
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = batch_sz } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = seq_len } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = embed_dim, .end = embed_dim * 2 } } },
        },
    };
    index_tuple_t tup_v = {
        .len = 3,
        .items = (tensor_index_t[]) {
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = batch_sz } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = seq_len } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = embed_dim * 2, .end = embed_dim * 3 } } },
        },
    };
    tensor* q = qkv->slice(qkv, tup_q, &err);
    tensor* k = qkv->slice(qkv, tup_k, &err);
    tensor* v = qkv->slice(qkv, tup_v, &err);

    tensor* q_cont = q->contiguous(q, &err);
    tensor* k_cont = k->contiguous(k, &err);
    tensor* v_cont = v->contiguous(v, &err);

    tensor* tmp = q; q = q_cont;
    tensor_free(tmp); free(tmp);
    tmp = k; k = k_cont;
    tensor_free(tmp); free(tmp);
    tmp = v; v = v_cont;
    tensor_free(tmp); free(tmp);

    tensor* q_view = q->view(
        q, 
        dynarray_create((int32_t[]){batch_sz, seq_len, module->num_heads, embed_dim / module->num_heads}, 4, TYPE_INT32, &err),
        &err
    );
    tensor* q_base = q;
    q = q_view->transpose(q_view, 1, 2, &err);
    tensor_free(q_view); free(q_view);
    tensor* k_view = k->view(
        k, 
        dynarray_create((int32_t[]){batch_sz, seq_len, module->num_heads, embed_dim / module->num_heads}, 4, TYPE_INT32, &err),
        &err
    );
    tensor* k_base = k;
    k = k_view->transpose(k_view, 1, 2, &err);
    tensor_free(k_view); free(k_view);
    tensor* v_view = v->view(
        v, 
        dynarray_create((int32_t[]){batch_sz, seq_len, module->num_heads, embed_dim / module->num_heads}, 4, TYPE_INT32, &err),
        &err
    );
    tensor* v_base = v;
    v = v_view->transpose(v_view, 1, 2, &err);
    tensor_free(v_view); free(v_view);

    float sqrt_q_shape_last = sqrtf((float)(embed_dim / module->num_heads));
    tensor* norm = tensor_create_scalar(&sqrt_q_shape_last, TYPE_FLOAT, &err);
    tensor* k_transposed = k->transpose(k, 2, 3, &err);
    tensor* mul_q_k = tensor_matmul(q, k_transposed, &err);
    tensor_free(q); free(q);
    tensor_free(k_transposed); free(k_transposed);
    tensor_free(k); free(k);
    tensor* raw_scores = tensor_divide(mul_q_k, norm, mul_q_k->type, &err);
    tensor_free(mul_q_k); free(mul_q_k);
    tensor_free(norm); free(norm);

    index_tuple_t tup_mask = {
        .len = 4,
        .items = (tensor_index_t[]) {
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = 1 } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = 1 } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = seq_len } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = seq_len } } },
        },
    };
    tensor* mask_slice = module->causal_mask->slice(module->causal_mask, tup_mask, &err);

    tensor* masked_raw_scores = tensor_add(raw_scores, mask_slice, raw_scores->type, &err);
    tensor_free(raw_scores); free(raw_scores);
    tensor_free(mask_slice); free(mask_slice);

    tensor* scores = softmax(masked_raw_scores, -1, &err);
    tensor_free(masked_raw_scores); free(masked_raw_scores);

    tensor* attention = tensor_matmul(scores, v, &err);
    tensor_free(scores); free(scores);
    tensor_free(v); free(v);
    tensor_free(q_base); free(q_base);
    tensor_free(k_base); free(k_base);
    tensor_free(v_base); free(v_base);

    tensor* attn_transposed = attention->transpose(attention, 1, 2, &err);
    tensor* attn_cont = attn_transposed->contiguous(attn_transposed, &err);
    tensor_free(attention); free(attention);
    tensor_free(attn_transposed); free(attn_transposed);

    tensor* attn_cont_view = attn_cont->view(attn_cont, dynarray_create((int32_t[]){batch_sz, seq_len, embed_dim}, 3, TYPE_INT32, &err), &err);
    attention = attn_cont_view->contiguous(attn_cont_view, &err);
    tensor_free(attn_cont_view); free(attn_cont_view);
    tensor_free(attn_cont); free(attn_cont);

    tensor* out = module->output_proj->module_base.forward(module->output_proj, attention);
    tensor_free(attention); free(attention);

    return out;
}

static void causal_self_attention_print(const void* self) {
    printf("Causal Self Attention\n");
}

static void feed_forward_free(void* self);

static void feed_forward_print(const void* self);

static tensor* feed_forward_forward(const void* self, const tensor* input);

feed_forward* feed_forward_create(linear_module* fc, linear_module* fc_proj, pml_err_t* err) {
    int32_t inp_sz = fc->weight->shape.get_at(&fc->weight->shape, 1).val.i;
    int32_t hidden_sz = fc->weight->shape.get_at(&fc->weight->shape, 0).val.i;
    int32_t hidden_proj_sz = fc_proj->weight->shape.get_at(&fc_proj->weight->shape, 1).val.i;
    int32_t out_sz = fc_proj->weight->shape.get_at(&fc_proj->weight->shape, 0).val.i;
    if (!(inp_sz == out_sz && hidden_proj_sz == hidden_sz)) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    feed_forward* module = (feed_forward*)malloc(sizeof(feed_forward));
    module->module_base = (module_iface){
        .destroy = feed_forward_free,
        .forward = feed_forward_forward,
        .print = feed_forward_print,
    };
    module->fc = fc;
    module->fc_proj = fc_proj;
    return module;
}

static void feed_forward_free(void* self) {
    feed_forward* module = (feed_forward*)self;
    module->fc->module_base.destroy(module->fc);
    module->fc_proj->module_base.destroy(module->fc_proj);
    free(module);
}

static void feed_forward_print(const void* self) {
    printf("Feed Forward Layer\n");
}

static tensor* feed_forward_forward(const void* self, const tensor* input) {
    feed_forward* module = (feed_forward*)self;
    pml_err_t err;
    
    tensor* fc_out = module->fc->module_base.forward(module->fc, input);

    tensor* gelu_out = gelu(fc_out, &err);
    tensor_free(fc_out); free(fc_out);

    tensor* out = module->fc_proj->module_base.forward(module->fc_proj, gelu_out);
    tensor_free(gelu_out); free(gelu_out);

    return out;
}

static void gpt_block_free(void* self);

static void gpt_block_print(const void* self);

static tensor* gpt_block_forward(const void* self, const tensor* input);

gpt_block* gpt_block_create(layernorm* ln_1, layernorm* ln_2, causal_self_attention* attn, feed_forward* ffn) {
    gpt_block* module = (gpt_block*)malloc(sizeof(gpt_block));
    module->module_base = (module_iface){
        .destroy = gpt_block_free,
        .forward = gpt_block_forward,
        .print = gpt_block_print,
    };
    module->ln_1 = ln_1;
    module->ln_2 = ln_2;
    module->attn = attn;
    module->ffn = ffn;
    return module;
}

static void gpt_block_free(void* self) {
    gpt_block* module = (gpt_block*)self;
    module->ln_1->module_base.destroy(module->ln_1);
    module->ln_2->module_base.destroy(module->ln_2);
    module->attn->module_base.destroy(module->attn);
    module->ffn->module_base.destroy(module->ffn);
    free(module);
}

static void gpt_block_print(const void* self) {
    printf("GPT Block\n");
}

static tensor* gpt_block_forward(const void* self, const tensor* input) {
    gpt_block* module = (gpt_block*)self;
    pml_err_t err;

    tensor* ln_1_out = module->ln_1->module_base.forward(module->ln_1, input);
    tensor* attn_out = module->attn->module_base.forward(module->attn, ln_1_out);
    tensor* inp_skip = tensor_add(input, attn_out, input->type, &err);
    tensor_free(ln_1_out); free(ln_1_out);
    tensor_free(attn_out); free(attn_out);

    tensor* ln_2_out = module->ln_2->module_base.forward(module->ln_2, inp_skip);
    tensor* ffn_out = module->ffn->module_base.forward(module->ffn, ln_2_out);
    tensor* out = tensor_add(inp_skip, ffn_out, inp_skip->type, &err);
    tensor_free(ln_2_out); free(ln_2_out);
    tensor_free(ffn_out); free(ffn_out);

    return out;
}