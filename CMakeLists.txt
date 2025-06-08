cmake_minimum_required(VERSION 3.31)
project(primitiveml LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(pml src/dynarray.c src/tensor.c)

file(GLOB MAIN_SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.c)
add_executable(primitiveml ${MAIN_SRC_FILES})
target_link_libraries(primitiveml PRIVATE pml)


enable_testing()

add_subdirectory(tests)
