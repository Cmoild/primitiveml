#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <tensor_index.h>


int main(){
    pml_err_t err = PML_OK;

    dynarray shape = dynarray_create((int32_t[]){2,2,3}, 3, TYPE_INT32, &err);
    tensor* t = tensor_create((int32_t[]){1,2,3,4,5,6,7,8,9,10,11,12}, 12, TYPE_INT32, 3, shape, &err);
    index_tuple_t tup = {
        .len = 3,
        .items = (tensor_index_t[]) {
            { .type = IDX_INT, .value = { .index = 0 } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = 2 } } },
            { .type = IDX_SLICE, .value = { .slice = { .start = 0, .end = 1 } } },
        },
    };
    tensor* sl = t->slice(t, tup, &err);
    sl->print(sl);

    return 0;
}
