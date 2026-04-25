set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(pml_core LANGUAGES C CXX)

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

find_package(OpenMP REQUIRED)

add_library(pml_core)

target_sources(pml_core
    PRIVATE
        src/tensor.cpp
    PUBLIC
        FILE_SET CXX_MODULES TYPE CXX_MODULES FILES
            pml.cppm
            tensor.cppm
            operations.cppm
            matmul.cppm
            slice.cppm
            storage.cppm
            exceptions.cppm
            tensor_iterator.cppm
)

target_compile_features(pml_core PUBLIC cxx_std_26)

target_include_directories(pml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_NUMPY_TESTS)
    set_target_properties(pml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
