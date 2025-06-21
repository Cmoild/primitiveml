#ifndef CONTAINER_TYPE_H
#define CONTAINER_TYPE_H

#include <inttypes.h>
#include <error_handling.h>

typedef enum container_type_t {
    TYPE_FLOAT,
    TYPE_INT32,
} container_type_t;

typedef union value_t {
    int32_t i;
    float f;
} value_t;

typedef struct result_t {
    container_type_t type;
    value_t val;
    pml_err_t err;
} result_t;

#endif // CONTAINER_TYPE_H