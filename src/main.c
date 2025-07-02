#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>
#include <tensor_index.h>
#include <embedding.h>


int main(){
    pml_err_t err = PML_OK;

    dynarray shape1 = dynarray_create((int32_t[]){1,4}, 2, TYPE_INT32, &err);
    tensor* t1 = tensor_create((int[]){5, 1, 0, 2}, 4, TYPE_INT32, 2, shape1, &err);

    tensor* embed_weight = tensor_create(
        (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.11f, 0.12f, 0.13f,},
        12, TYPE_FLOAT, 2, dynarray_create((int[]){6, 2}, 2, TYPE_INT32, &err), &err
    );
    embedding_module* embed = embedding_module_create(embed_weight, &err);
    tensor* out = embed->module_base.forward(embed, t1);
    out->print(out);
    return 0;
}
