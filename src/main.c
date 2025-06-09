#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>


int main(){
    float raw_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t raw_shape[] = {1, 2, 2, 3};
    pml_err_t err = PML_OK;
    dynarray shape = dynarray_create(raw_shape, 4, TYPE_INT32, &err);
    printf("Shape: ");
    shape.print(&shape);
    tensor* tensor = tensor_create(raw_data, 12, TYPE_FLOAT, 4, shape, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tensor->print(tensor);
    tensor_free(tensor);
    free(tensor);
    return 0;
}
