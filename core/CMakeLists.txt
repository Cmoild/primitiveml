project(pml_core LANGUAGES C)

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenMP REQUIRED)

add_library(pml_core 
    src/dynarray.c 
    src/tensor.c
    src/tensor_iterator.c
)

target_include_directories(pml_core 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${BLAS_INCLUDE_DIRS}
)

target_link_libraries(pml_core 
    PRIVATE m
    PRIVATE OpenMP::OpenMP_C
)
