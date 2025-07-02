#include <embedding.h>
#include <module.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


static void embedding_module_free(void* self);

static void embedding_module_print(const void* self);

static tensor* embedding_module_forward(const void* self, const tensor* input);

embedding_module* embedding_module_create(tensor* weight, pml_err_t* err) {
    if (weight->n_dim != 2 || weight->type == TYPE_INT32) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    if (!tensor_is_contiguous(weight, err)) {
        *err = PML_INCORRECT_INPUT;
        return NULL;
    }
    embedding_module* module = (embedding_module*)malloc(sizeof(embedding_module));
    module->weight = weight;
    module->module_base = (module_iface){
        .destroy = embedding_module_free,
        .forward = embedding_module_forward,
        .print = embedding_module_print,
    };
    module->num_embeddings = (size_t)weight->shape.get_at(&weight->shape, 0).val.i;
    module->embedding_dim = (size_t)weight->shape.get_at(&weight->shape, 1).val.i;
    *err = PML_OK;
    return module;
}

static void embedding_module_free(void* self) {
    embedding_module* module = (embedding_module*)self;
    tensor_free(module->weight);
    free(module->weight);
    free(module);
}

static tensor* embedding_module_forward(const void* self, const tensor* input) {
    embedding_module* module = (embedding_module*)self;
    pml_err_t err;
    err = PML_OK;
    if (input->type != TYPE_INT32 && !tensor_is_contiguous(input, &err)) {
        return NULL;
    }
    float* res_data = (float*)malloc(sizeof(float) * module->embedding_dim * input->data_num_elems);
    dynarray res_shape = dynarray_clone(&input->shape, &err);
    res_shape.insert_at(&res_shape, res_shape._size, &module->embedding_dim);
    int32_t* inp_data = (int32_t*)input->data;
    for (size_t i = 0; i < input->data_num_elems; i++) {
        if ((size_t)inp_data[i] >= module->num_embeddings) {
            free(res_data);
            dynarray_free(&res_shape);
            return NULL;
        }
        memcpy(
            res_data + i * module->embedding_dim, 
            (float*)module->weight->data + (size_t)inp_data[i] * module->embedding_dim, 
            module->embedding_dim * sizeof(float)
        );
    }
    tensor* output = tensor_create_without_copy(
        res_data, module->embedding_dim * input->data_num_elems, 
        TYPE_FLOAT, res_shape._size, res_shape, &err
    );
    return output;
}

static void embedding_module_print(const void* self) {
    printf("Embedding\n");
}