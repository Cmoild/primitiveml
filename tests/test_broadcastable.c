#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include "tensor.h"
#include "dynarray.h"
#include "error_handling.h"
#include "container_type.h" 

// Helper function to create a tensor for testing
tensor* create_test_tensor(const int32_t* data, size_t data_len, const int32_t* shape_data, size_t n_dimensions, pml_err_t* err) {
    dynarray shape = dynarray_create((void*)shape_data, n_dimensions, TYPE_INT32, err);
    if (*err != PML_OK) {
        return NULL;
    }
    tensor* t = tensor_create((void*)data, data_len, TYPE_INT32, n_dimensions, shape, err);
    return t;
}

// Test Case 1: Identical shapes should be broadcastable
void test_identical_shapes() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4};
    int32_t shape1[] = {2, 2};
    tensor* t1 = create_test_tensor(data1, 4, shape1, 2, &err);
    assert(err == PML_OK);

    int32_t data2[] = {5, 6, 7, 8};
    int32_t shape2[] = {2, 2};
    tensor* t2 = create_test_tensor(data2, 4, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 1 (Identical Shapes): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 2: Scalar and a tensor
void test_scalar_and_tensor() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {10};
    int32_t shape1[] = {1}; // Scalar
    tensor* t1 = create_test_tensor(data1, 1, shape1, 1, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3, 4, 5, 6};
    int32_t shape2[] = {2, 3};
    tensor* t2 = create_test_tensor(data2, 6, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 2 (Scalar and Tensor): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 3: Different number of dimensions, right-aligned 1s
void test_different_dims_right_aligned_ones() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3};
    int32_t shape1[] = {3};
    tensor* t1 = create_test_tensor(data1, 3, shape1, 1, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t shape2[] = {3, 3};
    tensor* t2 = create_test_tensor(data2, 9, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 3 (Different Dims, Right-aligned 1s): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 4: Different number of dimensions, right-aligned 1s (swapped)
void test_different_dims_right_aligned_ones_swapped() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t shape1[] = {3, 3};
    tensor* t1 = create_test_tensor(data1, 9, shape1, 2, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3};
    int32_t shape2[] = {3};
    tensor* t2 = create_test_tensor(data2, 3, shape2, 1, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 4 (Different Dims, Right-aligned 1s - Swapped): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}


// Test Case 5: Broadcastable with 1s in shape
void test_broadcastable_with_ones() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4};
    int32_t shape1[] = {4, 1};
    tensor* t1 = create_test_tensor(data1, 4, shape1, 2, &err);
    assert(err == PML_OK);

    int32_t data2[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int32_t shape2[] = {4, 2};
    tensor* t2 = create_test_tensor(data2, 8, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 5 (Broadcastable with 1s): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 6: Broadcastable with 1s in shape (swapped)
void test_broadcastable_with_ones_swapped() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int32_t shape1[] = {4, 2};
    tensor* t1 = create_test_tensor(data1, 8, shape1, 2, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3, 4};
    int32_t shape2[] = {4, 1};
    tensor* t2 = create_test_tensor(data2, 4, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 6 (Broadcastable with 1s - Swapped): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 7: Non-broadcastable due to incompatible dimensions
void test_non_broadcastable_incompatible_dims() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4};
    int32_t shape1[] = {2, 2};
    tensor* t1 = create_test_tensor(data1, 4, shape1, 2, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3, 4, 5, 6};
    int32_t shape2[] = {2, 3}; // Mismatch at last dimension
    tensor* t2 = create_test_tensor(data2, 6, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK); // Should still return PML_OK, just broadcastable is false
    assert(broadcastable == false);

    printf("Test Case 7 (Non-Broadcastable - Incompatible Dims): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 8: Non-broadcastable due to different number of dimensions and no 1s
void test_non_broadcastable_different_dims_no_ones() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2};
    int32_t shape1[] = {2};
    tensor* t1 = create_test_tensor(data1, 2, shape1, 1, &err);
    assert(err == PML_OK);

    int32_t data2[] = {1, 2, 3, 4, 5, 6};
    int32_t shape2[] = {2, 3}; // Dimensions don't align from right
    tensor* t2 = create_test_tensor(data2, 6, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == false);

    printf("Test Case 8 (Non-Broadcastable - Different Dims, No 1s): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 9: Broadcasting with a 3D tensor and a 1D tensor
void test_3d_and_1d_broadcastable() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t shape1[] = {2, 2, 3};
    tensor* t1 = create_test_tensor(data1, 12, shape1, 3, &err);
    assert(err == PML_OK);

    int32_t data2[] = {10, 20, 30};
    int32_t shape2[] = {3};
    tensor* t2 = create_test_tensor(data2, 3, shape2, 1, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 9 (3D and 1D Broadcastable): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 10: Broadcasting with a 3D tensor and a 2D tensor (compatible)
void test_3d_and_2d_broadcastable() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t shape1[] = {2, 2, 3};
    tensor* t1 = create_test_tensor(data1, 12, shape1, 3, &err);
    assert(err == PML_OK);

    int32_t data2[] = {10, 20, 30, 40, 50, 60};
    int32_t shape2[] = {2, 3};
    tensor* t2 = create_test_tensor(data2, 6, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == true);

    printf("Test Case 10 (3D and 2D Broadcastable): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}

// Test Case 11: Non-broadcastable with 3D and 2D tensor (incompatible)
void test_3d_and_2d_non_broadcastable() {
    pml_err_t err = PML_OK;
    int32_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t shape1[] = {2, 2, 3};
    tensor* t1 = create_test_tensor(data1, 12, shape1, 3, &err);
    assert(err == PML_OK);

    int32_t data2[] = {10, 20, 30, 40};
    int32_t shape2[] = {2, 2}; // Mismatch at last dimension
    tensor* t2 = create_test_tensor(data2, 4, shape2, 2, &err);
    assert(err == PML_OK);

    bool broadcastable = tensor_shapes_broadcastable(t1, t2, &err);
    assert(err == PML_OK);
    assert(broadcastable == false);

    printf("Test Case 11 (3D and 2D Non-Broadcastable): PASSED\n");
    tensor_free(t1);
    tensor_free(t2);
}


int main() {
    printf("Running tensor broadcasting tests...\n");

    test_identical_shapes();
    test_scalar_and_tensor();
    test_different_dims_right_aligned_ones();
    test_different_dims_right_aligned_ones_swapped();
    test_broadcastable_with_ones();
    test_broadcastable_with_ones_swapped();
    test_non_broadcastable_incompatible_dims();
    test_non_broadcastable_different_dims_no_ones();
    test_3d_and_1d_broadcastable();
    test_3d_and_2d_broadcastable();
    test_3d_and_2d_non_broadcastable();

    printf("All tensor broadcasting tests completed.\n");

    return 0;
}