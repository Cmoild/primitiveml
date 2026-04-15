cmake_minimum_required(VERSION 3.14)

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

add_executable(test_broadcasted_strides test_broadcasted_strides.cpp)

target_link_libraries(test_broadcasted_strides
    PRIVATE
        pml_core
        Catch2::Catch2WithMain
)

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
endif()
