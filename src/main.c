#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>
#include <tensor_index.h>


int main(){
    pml_err_t err = PML_OK;

    dynarray shape1 = dynarray_create((int32_t[]){5,2}, 2, TYPE_INT32, &err);
    tensor* t1 = tensor_create((int[]){1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 10, TYPE_INT32, 2, shape1, &err);

    tensor* t1_transposed = t1->transpose(t1, 0, 1, &err);

    t1_transposed->print(t1_transposed);
    printf("Is contiguous: %d\n", tensor_is_contiguous(t1_transposed, &err));

    tensor* t1t_view = t1_transposed->view(t1_transposed, dynarray_create((int32_t[]){10}, 1, TYPE_INT32, &err), &err);

    t1t_view->print(t1t_view);
    printf("Is contiguous: %d\n", tensor_is_contiguous(t1t_view, &err));

    tensor* t1tc = t1_transposed->contiguous(t1_transposed, &err);
    printf("Is contiguous: %d\n", tensor_is_contiguous(t1tc, &err));
    tensor* t1tc_view = t1tc->view(t1tc, dynarray_create((int32_t[]){10}, 1, TYPE_INT32, &err), &err);

    t1tc_view->print(t1tc_view);
    printf("Is contiguous: %d\n", tensor_is_contiguous(t1tc_view, &err));
    return 0;
}
