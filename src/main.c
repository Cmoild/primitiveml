#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>
#include <tensor_index.h>
#include <embedding.h>
#include <gpt_modules.h>
#include <layernorm.h>

#define MODEL_DIR "../nanogpt/model/"

typedef struct {
    float* qkv_proj_weight;
    size_t qkv_proj_weight_size;

    float* qkv_proj_bias;
    size_t qkv_proj_bias_size;

    float* out_proj_weight;
    size_t out_proj_weight_size;

    float* out_proj_bias;
    size_t out_proj_bias_size;

    float* mask;
    size_t mask_size;

    float* ln1_weight;
    size_t ln1_weight_size;

    float* ln1_bias;
    size_t ln1_bias_size;

    float* ln2_weight;
    size_t ln2_weight_size;

    float* ln2_bias;
    size_t ln2_bias_size;

    float* ffn_in_weight;
    size_t ffn_in_weight_size;

    float* ffn_in_bias;
    size_t ffn_in_bias_size;

    float* ffn_out_weight;
    size_t ffn_out_weight_size;

    float* ffn_out_bias;
    size_t ffn_out_bias_size;
} gpt_block_float;

void free_block(gpt_block_float* block) {
    if (!block)
        return;

    free(block->qkv_proj_weight);
    free(block->qkv_proj_bias);
    free(block->out_proj_weight);
    free(block->out_proj_bias);
    free(block->mask);
    free(block->ln1_weight);
    free(block->ln1_bias);
    free(block->ln2_weight);
    free(block->ln2_bias);
    free(block->ffn_in_weight);
    free(block->ffn_in_bias);
    free(block->ffn_out_weight);
    free(block->ffn_out_bias);

    free(block);
}

static float* load_bin_file(const char* path, size_t* num_elements_out) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    if (size % sizeof(float) != 0) {
        fprintf(stderr, "File size is not a multiple of sizeof(float): %s\n", path);
        fclose(file);
        return NULL;
    }

    size_t num_elements = size / sizeof(float);
    float* data = (float*)malloc(size);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for file: %s\n", path);
        fclose(file);
        return NULL;
    }

    fread(data, sizeof(float), num_elements, file);
    fclose(file);

    if (num_elements_out) {
        *num_elements_out = num_elements;
    }

    return data;
}

static gpt_block* load_block(size_t idx, size_t cont_wind_sz, size_t embed_dim, size_t n_heads) {
    gpt_block_float* block = (gpt_block_float*)malloc(sizeof(gpt_block_float));
    if (!block) {
        fprintf(stderr, "Failed to allocate gpt_block\n");
        return NULL;
    }

    char path[512];
    size_t dummy;

    // Load qkv_proj weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/qkv_proj/weight.bin", idx);
    block->qkv_proj_weight = load_bin_file(path, &dummy);
    block->qkv_proj_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/qkv_proj/bias.bin", idx);
    block->qkv_proj_bias = load_bin_file(path, &dummy);
    block->qkv_proj_bias_size = dummy;

    // Load out_proj weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/out_proj/weight.bin", idx);
    block->out_proj_weight = load_bin_file(path, &dummy);
    block->out_proj_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/out_proj/bias.bin", idx);
    block->out_proj_bias = load_bin_file(path, &dummy);
    block->out_proj_bias_size = dummy;

    // Load attention mask
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/mask.bin", idx);
    block->mask = load_bin_file(path, &dummy);
    block->mask_size = dummy;

    // Load layer norm 1 weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ln1/weight.bin", idx);
    block->ln1_weight = load_bin_file(path, &dummy);
    block->ln1_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ln1/bias.bin", idx);
    block->ln1_bias = load_bin_file(path, &dummy);
    block->ln1_bias_size = dummy;

    // Load layer norm 2 weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ln2/weight.bin", idx);
    block->ln2_weight = load_bin_file(path, &dummy);
    block->ln2_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ln2/bias.bin", idx);
    block->ln2_bias = load_bin_file(path, &dummy);
    block->ln2_bias_size = dummy;

    // Load FFN input weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ffn/in/weight.bin", idx);
    block->ffn_in_weight = load_bin_file(path, &dummy);
    block->ffn_in_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ffn/in/bias.bin", idx);
    block->ffn_in_bias = load_bin_file(path, &dummy);
    block->ffn_in_bias_size = dummy;

    // Load FFN output weights and bias
    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ffn/out/weight.bin", idx);
    block->ffn_out_weight = load_bin_file(path, &dummy);
    block->ffn_out_weight_size = dummy;

    snprintf(path, sizeof(path), MODEL_DIR "blocks/%zu/ffn/out/bias.bin", idx);
    block->ffn_out_bias = load_bin_file(path, &dummy);
    block->ffn_out_bias_size = dummy;

    pml_err_t err;

    tensor* mask = tensor_create(
        block->mask, block->mask_size, TYPE_FLOAT, 4,
        dynarray_create((int32_t[]){1, 1, (int32_t)cont_wind_sz, (int32_t)cont_wind_sz}, 4,
                        TYPE_INT32, &err),
        &err);

    tensor* qkv_proj_weight =
        tensor_create(block->qkv_proj_weight, block->qkv_proj_weight_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){block->qkv_proj_weight_size / embed_dim, embed_dim},
                                      2, TYPE_INT32, &err),
                      &err);
    tensor* qkv_proj_bias = tensor_create(
        block->qkv_proj_bias, block->qkv_proj_bias_size, TYPE_FLOAT, 1,
        dynarray_create((int[]){block->qkv_proj_bias_size}, 1, TYPE_INT32, &err), &err);
    linear_module* qkv_proj = linear_module_create(qkv_proj_weight, qkv_proj_bias, &err);

    tensor* out_proj_weight =
        tensor_create(block->out_proj_weight, block->out_proj_weight_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){embed_dim, embed_dim}, 2, TYPE_INT32, &err), &err);
    tensor* out_proj_bias =
        tensor_create(block->out_proj_bias, block->out_proj_bias_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    linear_module* out_proj = linear_module_create(out_proj_weight, out_proj_bias, &err);

    causal_self_attention* mha =
        causal_self_attention_create(qkv_proj, out_proj, mask, n_heads, embed_dim, &err);

    tensor* ln1_weight =
        tensor_create(block->ln1_weight, block->ln1_weight_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    tensor* ln1_bias =
        tensor_create(block->ln1_bias, block->ln1_bias_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    tensor* ln2_weight =
        tensor_create(block->ln2_weight, block->ln2_weight_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    tensor* ln2_bias =
        tensor_create(block->ln2_bias, block->ln2_bias_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    layernorm* ln1 = layernorm_create(ln1_weight, ln1_bias, 1e-5, &err);
    layernorm* ln2 = layernorm_create(ln2_weight, ln2_bias, 1e-5, &err);

    tensor* ffn_in_weight =
        tensor_create(block->ffn_in_weight, block->ffn_in_weight_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){block->ffn_in_weight_size / embed_dim, embed_dim}, 2,
                                      TYPE_INT32, &err),
                      &err);
    tensor* ffn_in_bias =
        tensor_create(block->ffn_in_bias, block->ffn_in_bias_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){block->ffn_in_bias_size}, 1, TYPE_INT32, &err), &err);
    linear_module* ffn_in = linear_module_create(ffn_in_weight, ffn_in_bias, &err);

    tensor* ffn_out_weight =
        tensor_create(block->ffn_out_weight, block->ffn_out_weight_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){embed_dim, block->ffn_out_weight_size / embed_dim}, 2,
                                      TYPE_INT32, &err),
                      &err);
    tensor* ffn_out_bias = tensor_create(
        block->ffn_out_bias, block->ffn_out_bias_size, TYPE_FLOAT, 1,
        dynarray_create((int[]){block->ffn_out_bias_size}, 1, TYPE_INT32, &err), &err);
    linear_module* ffn_out = linear_module_create(ffn_out_weight, ffn_out_bias, &err);

    feed_forward* ffn = feed_forward_create(ffn_in, ffn_out, &err);

    gpt_block* block_out = gpt_block_create(ln1, ln2, mha, ffn);

    free_block(block);

    return block_out;
}

typedef struct {
    float* wte;
    size_t wte_size;

    float* wpe;
    size_t wpe_size;

    float* ln_weight;
    size_t ln_weight_size;

    float* ln_bias;
    size_t ln_bias_size;

    float* fc_weight;
    size_t fc_weight_size;

    float* fc_bias;
    size_t fc_bias_size;
} gpt_model_float;

void free_model_weights(gpt_model_float* model) {
    if (!model)
        return;

    free(model->wte);
    free(model->wpe);
    free(model->ln_weight);
    free(model->ln_bias);
    free(model->fc_weight);
    free(model->fc_bias);

    free(model);
}

static gpt_model* load_model(size_t num_blocks, size_t cont_wind_sz, size_t embed_dim,
                             size_t n_heads, size_t vocab_sz) {
    gpt_model_float* model = (gpt_model_float*)malloc(sizeof(gpt_model_float));
    if (!model) {
        fprintf(stderr, "Error: could not allocate model\n");
        return NULL;
    }

    char path[512];

    snprintf(path, sizeof(path), MODEL_DIR "wte.bin");
    model->wte = load_bin_file(path, &model->wte_size);

    snprintf(path, sizeof(path), MODEL_DIR "wpe.bin");
    model->wpe = load_bin_file(path, &model->wpe_size);

    snprintf(path, sizeof(path), MODEL_DIR "ln/weight.bin");
    model->ln_weight = load_bin_file(path, &model->ln_weight_size);

    snprintf(path, sizeof(path), MODEL_DIR "ln/bias.bin");
    model->ln_bias = load_bin_file(path, &model->ln_bias_size);

    snprintf(path, sizeof(path), MODEL_DIR "fc/weight.bin");
    model->fc_weight = load_bin_file(path, &model->fc_weight_size);

    snprintf(path, sizeof(path), MODEL_DIR "fc/bias.bin");
    model->fc_bias = load_bin_file(path, &model->fc_bias_size);

    pml_err_t err;

    tensor* wte_weight =
        tensor_create(model->wte, model->wte_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){vocab_sz, embed_dim}, 2, TYPE_INT32, &err), &err);
    embedding_module* wte = embedding_module_create(wte_weight, &err);

    tensor* wpe_weight =
        tensor_create(model->wpe, model->wpe_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){cont_wind_sz, embed_dim}, 2, TYPE_INT32, &err), &err);
    embedding_module* wpe = embedding_module_create(wpe_weight, &err);

    tensor* ln_weight =
        tensor_create(model->ln_weight, model->ln_weight_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    tensor* ln_bias = tensor_create(model->ln_bias, model->ln_bias_size, TYPE_FLOAT, 1,
                                    dynarray_create((int[]){embed_dim}, 1, TYPE_INT32, &err), &err);
    layernorm* ln = layernorm_create(ln_weight, ln_bias, 1e-5, &err);

    tensor* fc_weight =
        tensor_create(model->fc_weight, model->fc_weight_size, TYPE_FLOAT, 2,
                      dynarray_create((int[]){vocab_sz, embed_dim}, 2, TYPE_INT32, &err), &err);
    tensor* fc_bias =
        tensor_create(model->fc_bias, model->fc_bias_size, TYPE_FLOAT, 1,
                      dynarray_create((int[]){model->fc_bias_size}, 1, TYPE_INT32, &err), &err);
    linear_module* fc = linear_module_create(fc_weight, fc_bias, &err);

    gpt_block** blocks = (gpt_block**)malloc(sizeof(gpt_block*) * num_blocks);

    for (size_t i = 0; i < num_blocks; i++) {
        gpt_block* block = load_block(i, cont_wind_sz, embed_dim, n_heads);
        blocks[i] = block;
    }

    gpt_model* model_out = gpt_model_create(wte, wpe, blocks, num_blocks, ln, fc);

    free_model_weights(model);

    return model_out;
}

int argmax(tensor* probs) {
    pml_err_t err;
    index_tuple_t tup = {
        .len = 2,
        .items =
            (tensor_index_t[]){
                {.type = IDX_INT, .value = {.index = 0}},
                {.type = IDX_INT, .value = {.index = -1}},
            },
    };
    tensor* probs_last = probs->slice(probs, tup, &err);
    float* data = probs_last->data;
    float max = 0;
    int max_idx = 0;
    for (int i = 0; i < (int)probs_last->data_num_elems; i++) {
        if (data[i] > max) {
            max = data[i];
            max_idx = i;
        }
    }
    tensor_free(probs_last);
    free(probs_last);
    return max_idx;
}

void load_prompt(int* buf, const char* prompt, const size_t len) {
    for (size_t i = 0; i < len; i++) {
        buf[i] = prompt[i];
    }
}

void print_usage(const char* progname) {
    printf("Usage: %s [PROMPT] [NUM_OF_TOKENS_TO_GENERATE]\n", progname);
    printf("\n");
    printf("Arguments:\n");
    printf("  PROMPT                       Text prompt (required).\n");
    printf("  NUM_OF_TOKENS_TO_GENERATE    Positive integer (required).\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help                   Show this help message and exit.\n");
}

long parse_positive_long(const char* s, int* ok) {
    errno = 0;
    char* endptr = NULL;
    long val = strtol(s, &endptr, 10);
    if (errno != 0 || endptr == s || *endptr != '\0') {
        *ok = 0;
        return -1;
    }
    if (val <= 0) {
        *ok = 0;
        return -1;
    }
    *ok = 1;
    return val;
}

int main(int argc, char* argv[]) {
    const char* progname = argv[0];

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(progname);
            return 0;
        }
    }

    if (argc < 3) {
        fprintf(stderr, "Error: PROMPT and NUM_OF_TOKENS_TO_GENERATE are required.\n\n");
        print_usage(progname);
        return 1;
    }

    const char* prompt = argv[1];
    long num_tokens;

    if (argc >= 3) {
        int ok;
        num_tokens = parse_positive_long(argv[2], &ok);
        if (!ok) {
            fprintf(stderr,
                    "Error: NUM_OF_TOKENS_TO_GENERATE must be a positive integer (got '%s').\n",
                    argv[2]);
            return 2;
        }
    }

    pml_err_t err = PML_OK;

    gpt_model* model = load_model(2, 256, 128, 4, 256);

    int buf[256] = {0};
    load_prompt(buf, prompt, strlen(prompt));

    printf("%s", prompt);
    fflush(stdout);

    for (int i = strlen(prompt); i < num_tokens - strlen(prompt); i++) {
        tensor* inp = tensor_create(buf, i, TYPE_INT32, 2,
                                    dynarray_create((int[]){1, i}, 2, TYPE_INT32, &err), &err);
        tensor* out = model->module_base.forward(model, inp);
        tensor* probs = softmax(out, -1, &err);
        buf[i] = argmax(probs);
        printf("%c", buf[i]);
        fflush(stdout);

        tensor_free(inp);
        free(inp);
        tensor_free(out);
        free(out);
        tensor_free(probs);
        free(probs);
    }
    model->module_base.destroy(model);
    printf("\n");
    return 0;
}
