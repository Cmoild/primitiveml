cmake_minimum_required(VERSION 3.31)
project(benchmark LANGUAGES C CXX Fortran)

find_package(OpenMP REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

add_executable(gemm_benchmark
    ${CMAKE_SOURCE_DIR}/benchmarks/gemm_benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE pml_core m OpenBLAS::OpenBLAS)
