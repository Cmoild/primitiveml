#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t shape_init_raw[] = {2, 2, 3};
    dynarray shape_init = dynarray_create(shape_init_raw, 3, TYPE_INT32, &err);
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    tensor* tens = tensor_create(data, 12, TYPE_FLOAT, 3, shape_init, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tens->print(tens);
    
    tensor* new = tensor_axis_var(tens, 0, &err);

    new->print(new);

    tensor_free(tens);
    free(tens);
    tensor_free(new);
    free(new);
    return 0;
}
