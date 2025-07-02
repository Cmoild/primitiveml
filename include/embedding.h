#ifndef EMBEDDING_MODULE_H
#define EMBEDDING_MODULE_H

#include <module.h>
#include <error_handling.h>

typedef struct embedding_module {
    module_iface module_base;
    tensor* weight;
    size_t num_embeddings;
    size_t embedding_dim;
} embedding_module;

embedding_module* embedding_module_create(tensor* weight, pml_err_t* err);

#endif // EMBEDDING_MODULE_H