cmake_minimum_required(VERSION 3.31)
project(primitiveml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(pmlcompute REQUIRED)

add_library(primitiveml STATIC
    pml/src/tensor.cpp
)

target_include_directories(primitiveml
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/pml/include
)

function(set_warnings_and_build_flags target)
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g>
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/O2 /DNDEBUG /Zi>
            $<$<CONFIG:Debug>:/Od /Zi>
        )
    endif()
endfunction()

set_warnings_and_build_flags(primitiveml)

if(NOT WIN32)
    target_link_libraries(primitiveml PRIVATE m)
endif()

add_executable(pml_app
    examples/main.cpp
)

set_warnings_and_build_flags(pml_app)

target_link_libraries(pml_app PRIVATE primitiveml)

if (BUILD_CXX_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
