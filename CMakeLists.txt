cmake_minimum_required(VERSION 3.31)
project(primitiveml LANGUAGES C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -g0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

add_subdirectory(core)

# add_library(pml
#     src/linear.c
#     src/functional.c
#     src/conv2d.c
#     src/embedding.c
#     src/gpt_modules.c
#     src/layernorm.c
# )
#
# target_include_directories(pml
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
#         ${PROJECT_SOURCE_DIR}/core/include
# )
#
# target_link_libraries(pml PRIVATE m)

file(GLOB MAIN_SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_executable(primitiveml ${MAIN_SRC_FILES})
target_link_libraries(primitiveml PRIVATE pml_core m)

enable_testing()

add_subdirectory(tests)
