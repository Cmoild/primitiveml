project(pml_core LANGUAGES C CXX ASM)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g -fno-omit-frame-pointer -mavx2 -mfma")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer -mavx2 -mfma")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

find_package(OpenMP REQUIRED)

add_library(pml_core
    src/tensor.cpp
    src/exp_avx2.S
    src/exp_avx2_intrin.cpp
    src/cpu_flags.S
    src/gemm.c
    src/gemm_microkernel_avx2.S
    src/gemm_microkernel6x16_avx2.S
)

target_include_directories(pml_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${BLAS_INCLUDE_DIRS}
)

target_link_libraries(pml_core
    PRIVATE m
    PRIVATE OpenMP::OpenMP_C
)

if(BUILD_NUMPY_TESTS)
    set_target_properties(pml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
