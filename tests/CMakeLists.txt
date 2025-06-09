add_executable(test_dynarray test_dynarray.c)
target_link_libraries(test_dynarray PRIVATE pml_core)

add_executable(test_tensor test_tensor.c)
target_link_libraries(test_tensor PRIVATE pml_core)

add_executable(test_broadcastable test_broadcastable.c)
target_link_libraries(test_broadcastable PRIVATE pml_core)

add_test(NAME DynArrayTest COMMAND test_dynarray)
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME TensorBroadcastableTest COMMAND test_broadcastable)
