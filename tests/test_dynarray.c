#include <stdio.h>
#include <dynarray.h>


int basic_tests() {
    dynarray arr;
    int32_t src[] = {1, 2, 3, 4, 45, 1, 568};
    pml_err_t err = PML_OK;
    arr = dynarray_create(src, 7, TYPE_INT32, &err);
    if (err != PML_OK) {
        printf("create: Error code: %d\n", err);
        return 1;
    }
    printf("Array:\n");
    arr.print(&arr);
    result_t res = arr.get_at(&arr, 6);
    if (res.err != PML_OK) {
        printf("get_at: Error code: %d\n", res.err);
        return 1;
    }
    if (res.val.i != 568) {
        printf("get_at: wrong value: %d\n", res.val.i);
        return 1;
    }
    printf("Value: %d\n", res.val.i);
    res = arr.get_at(&arr, 7);
    if (res.err != PML_OUT_OF_BOUNDS) {
        printf("get_at: Error code: %d\n", res.err);
        return 1;
    }
    return 0;
}

int test_empty() {
    dynarray arr;
    pml_err_t err = PML_OK;
    arr = dynarray_create(NULL, 0, TYPE_INT32, &err);
    if (err != PML_OK) {
        printf("create empty: Error code: %d\n", err);
        return 1;
    }
    if (arr._capacity != 0 || arr._size != 0) {
        printf(
            "create empty: capacity or size are not equal to 0. Size: %ld, Capacity: %ld\n",
            arr._size, arr._capacity
        );
        return 1;
    }
    result_t res = arr.get_at(&arr, 7);
    if (res.err != PML_OUT_OF_BOUNDS) {
        printf("get_at: Error code: %d\n", res.err);
        return 1;
    }
    return 0;
}

int main(){
    int ret = basic_tests();
    if (ret != 0) {
        return 1;
    }
    ret = test_empty();
    if (ret != 0) {
        return 1;
    }
    return 0;
}