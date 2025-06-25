#include <conv2d.h>
#include <module.h>
#include <stdlib.h>
#include <stdio.h>
#include <tensor_index.h>


static void conv2d_module_free(void* self);

static void conv2d_module_print(const void* self);

static tensor* conv2d_module_forward(const void* self, const tensor* input);

conv2d_module* conv2d_module_create(tensor* weight, tensor* bias, size_t padding, pml_err_t* err) {
    if (weight->n_dim != 4 && bias->n_dim != 1) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    if (weight->type == TYPE_INT32 || bias->type == TYPE_INT32) {
        *err = PML_WRONG_TYPE;
        return NULL;
    }
    int32_t ker_h, ker_w;
    ker_h = weight->shape.get_at(&weight->shape, 2).val.i;
    ker_w = weight->shape.get_at(&weight->shape, 3).val.i;
    if ((ker_h != ker_w) || (ker_h % 2 != 1 && ker_w % 2 != 1)) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    if ((int32_t)padding / 2 > ker_h) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    conv2d_module* module = (conv2d_module*)malloc(sizeof(conv2d_module));
    module->module_base = (module_iface){
        .destroy = conv2d_module_free,
        .forward = conv2d_module_forward,
        .print = conv2d_module_print,
    };
    module->weight = weight;
    module->bias = bias;
    module->padding = padding;
    return module;
}

static void conv2d_module_free(void* self) {
    conv2d_module* module = (conv2d_module*)self;
    tensor_free(module->weight);
    free(module->weight);
    tensor_free(module->bias);
    free(module->bias);
}

static void convolve_2d_float_tensor(
    tensor* input, tensor* output, tensor* kernel, tensor* bias, size_t padding,
    int32_t in_height, int32_t in_width, int32_t out_height, int32_t out_width, int32_t kernel_sz,
    pml_err_t* err
) {
    if (input->n_dim != 2 && output->n_dim != 2 && kernel->n_dim != 2 && bias->n_dim != 0) {
        *err = PML_INCORRECT_INPUT;
        return;
    }
    // use iterator here

}

static tensor* conv2d_module_forward(const void* self, const tensor* input) {
    pml_err_t err = PML_OK;
    conv2d_module* module = (conv2d_module*)self;
    if (input->n_dim != 4) {
        return NULL;
    }
    int32_t batch_size, in_chan, in_height, in_width;
    batch_size = input->shape.get_at(&input->shape, 0).val.i;
    in_chan = input->shape.get_at(&input->shape, 1).val.i;
    in_height = input->shape.get_at(&input->shape, 2).val.i;
    in_width = input->shape.get_at(&input->shape, 3).val.i;
    int32_t out_chan, in_chan_module, kern_sz;
    out_chan = module->weight->shape.get_at(&module->weight->shape, 0).val.i;
    in_chan_module = module->weight->shape.get_at(&module->weight->shape, 1).val.i;
    kern_sz = module->weight->shape.get_at(&module->weight->shape, 2).val.i;
    if (in_chan != in_chan_module || kern_sz % 2 != 1) {
        return NULL;
    }
    int32_t out_height, out_width;
    out_height = in_height - (kern_sz - 1) + (int32_t)module->padding * 2;
    out_width = in_width - (kern_sz - 1) + (int32_t)module->padding * 2;
    dynarray output_shape = dynarray_create((int32_t[]){batch_size, out_chan, out_height, out_width}, 4, TYPE_INT32, &err);
    if (err != PML_OK) {
        return NULL;
    }
    tensor* output = tensor_create_zeros(TYPE_FLOAT, 4, output_shape, &err);
    if (err != PML_OK) {
        dynarray_free(&output_shape);
        return NULL;
    }
    for (int b = 0; b < batch_size; b++) {
        for (int out_ch = 0; out_ch < out_chan; out_ch++) {
            for (int in_ch = 0; in_ch < in_chan; in_ch++) {
                index_tuple_t tup_in = {
                    .len = 2,
                    .items = (tensor_index_t[]) {
                        { .type = IDX_INT, .value = { .index = b } },
                        { .type = IDX_INT, .value = { .index = in_ch } },
                    },
                };
                tensor* cur_in = input->slice(input, tup_in, &err);
                if (err != PML_OK) {
                    tensor_free(output);
                    free(output);
                    return NULL;
                }
                index_tuple_t tup_out = {
                    .len = 2,
                    .items = (tensor_index_t[]) {
                        { .type = IDX_INT, .value = { .index = b } },
                        { .type = IDX_INT, .value = { .index = out_ch } },
                    },
                };
                tensor* cur_out = output->slice(output, tup_out, &err);
                if (err != PML_OK) {
                    tensor_free(output);
                    free(output);
                    tensor_free(cur_in);
                    free(cur_in);
                    return NULL;
                }
                index_tuple_t tup_ker = {
                    .len = 2,
                    .items = (tensor_index_t[]) {
                        { .type = IDX_INT, .value = { .index = out_ch } },
                        { .type = IDX_INT, .value = { .index = in_ch } },
                    },
                };
                tensor* cur_kernel = module->weight->slice(module->weight, tup_ker, &err);
                if (err != PML_OK) {
                    tensor_free(output);
                    free(output);
                    tensor_free(cur_in);
                    free(cur_in);
                    tensor_free(cur_out);
                    free(cur_out);
                    return NULL;
                }
                index_tuple_t tup_bias = {
                    .len = 1,
                    .items = (tensor_index_t[]) {
                        { .type = IDX_INT, .value = { .index = out_ch } },
                    },
                };
                tensor* cur_bias = module->bias->slice(module->bias, tup_bias, &err);
                if (err != PML_OK) {
                    tensor_free(output);
                    free(output);
                    tensor_free(cur_in);
                    free(cur_in);
                    tensor_free(cur_out);
                    free(cur_out);
                    tensor_free(cur_kernel);
                    free(cur_kernel);
                    return NULL;
                }
                convolve_2d_float_tensor(
                    cur_in, cur_out, cur_kernel, cur_bias, module->padding,
                    in_height, in_width, out_height, out_width, kern_sz,
                    &err
                );
                tensor_free(cur_in);
                free(cur_in);
                tensor_free(cur_out);
                free(cur_out);
                tensor_free(cur_kernel);
                free(cur_kernel);
                tensor_free(cur_bias);
                free(cur_bias);
                if (err != PML_OK) {
                    tensor_free(output);
                    free(output);
                    return NULL;
                }
            }
        }
    }
    return output;
}

static void conv2d_module_print(const void* self) {
    printf("Conv2d\n");
}