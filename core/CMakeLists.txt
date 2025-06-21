project(pml_core LANGUAGES C)

add_library(pml_core 
    src/dynarray.c 
    src/tensor.c
    src/tensor_iterator.c
)

target_include_directories(pml_core 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pml_core PRIVATE m)
