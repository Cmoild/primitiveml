project(primitiveml LANGUAGES C CUDA HIP)

set(CMAKE_C_STANDARD 23)

include_directories(include)

add_executable(primitiveml
    src/*
    main.cpp
)

