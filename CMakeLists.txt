cmake_minimum_required(VERSION 3.31)
project(primitiveml LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.c)

add_executable(primitiveml ${SRC_FILES})
