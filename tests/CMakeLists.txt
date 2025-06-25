add_executable(test_dynarray test_dynarray.c)
target_link_libraries(test_dynarray PRIVATE pml_core)

add_executable(test_tensor test_tensor.c)
target_link_libraries(test_tensor PRIVATE pml_core)

add_executable(test_broadcastable test_broadcastable.c)
target_link_libraries(test_broadcastable PRIVATE pml_core)

add_executable(test_tensor_iterator test_tensor_iterator.c)
target_link_libraries(test_tensor_iterator PRIVATE pml_core)

add_executable(test_matmul test_matmul.c)
target_link_libraries(test_matmul PRIVATE pml_core)

add_executable(test_tensor_index test_tensor_index.c)
target_link_libraries(test_tensor_index PRIVATE pml_core)

add_test(NAME DynArrayTest COMMAND test_dynarray)
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME TensorBroadcastableTest COMMAND test_broadcastable)
add_test(NAME TensorIteratorTest COMMAND test_tensor_iterator)
add_test(NAME MatMulTest COMMAND test_matmul)
add_test(NAME TensorIndexTest COMMAND test_tensor_index)
