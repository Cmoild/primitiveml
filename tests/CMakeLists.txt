cmake_minimum_required(VERSION 3.31)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.14.0
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_broadcastable test_broadcastable.cpp)

target_link_libraries(test_broadcastable
    PRIVATE
        pml_core
        Catch2::Catch2WithMain
)

target_compile_features(test_broadcastable PUBLIC cxx_std_26)

add_executable(test_broadcasted_strides test_broadcasted_strides.cpp)

target_link_libraries(test_broadcasted_strides
    PRIVATE
        pml_core
        Catch2::Catch2WithMain
)

target_compile_features(test_broadcasted_strides PUBLIC cxx_std_26)

include(CTest)
include(Catch)

catch_discover_tests(test_broadcastable)
catch_discover_tests(test_broadcasted_strides)

if(BUILD_NUMPY_TESTS)
    add_library(pmltest SHARED numpy_c_api.cpp)

    target_link_libraries(pmltest
        PRIVATE
            pml_core
    )

    target_compile_features(pmltest PUBLIC cxx_std_26)
endif()
