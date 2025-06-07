add_executable(test_dynarray test_dynarray.c)

target_link_libraries(test_dynarray PRIVATE pml)

add_test(NAME DynArrayTest COMMAND test_dynarray)
