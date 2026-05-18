cmake_minimum_required(VERSION 3.31)
project(pml_compute LANGUAGES C ASM)

if (MSVC)
    message(FATAL_ERROR
        "MSVC is not supported by this project. Use clang-cl on Windows.")
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(pml_compute SHARED IMPORTED)

target_compile_options(pml_compute PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(primitiveml PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    )
endif()
