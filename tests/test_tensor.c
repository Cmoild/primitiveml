#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>


int main() {
    int32_t raw_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t raw_shape[] = {2, 2, 3};
    pml_err_t err = PML_OK;
    dynarray shape = dynarray_create(raw_shape, 3, TYPE_INT32, &err);
    printf("Shape: ");
    shape.print(&shape);
    tensor* tensor = tensor_create(raw_data, 12, TYPE_INT32, 3, shape, &err);
    if (err != PML_OK) {
        printf("Error: %d\n", err);
        return 1;
    }
    tensor->print(tensor);
    return 0;
}