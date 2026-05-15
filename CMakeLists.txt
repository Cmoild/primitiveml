cmake_minimum_required(VERSION 3.31)
project(primitiveml LANGUAGES C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mavx2 -mfma -g -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -mavx2 -mfma -g -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

find_package(OpenMP REQUIRED)

add_subdirectory(core)

file(GLOB MAIN_SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_executable(primitiveml ${MAIN_SRC_FILES})
target_link_libraries(primitiveml PRIVATE pml_core)

if(NOT WIN32)
    target_link_libraries(primitiveml PRIVATE m)
endif()

if (BUILD_CXX_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
