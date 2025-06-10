#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    pml_err_t err = PML_OK;
    int32_t raw_shape[] = {5, 1, 12};
    dynarray shape = dynarray_create(raw_shape, 3, TYPE_INT32, &err);
    tensor* tensor = tensor_create_zeros(TYPE_INT32, 3, shape, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tensor->print(tensor);
    
    tensor_free(tensor);
    free(tensor);
    return 0;
}
