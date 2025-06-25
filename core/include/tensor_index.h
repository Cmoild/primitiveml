#ifndef TENSOR_INDEX_H
#define TENSOR_INDEX_H

#include <inttypes.h>
#include <stddef.h>

typedef enum {
    IDX_SLICE,
    IDX_INT,
} index_type_t;

typedef struct {
    int32_t start;
    int32_t end;
} index_slice_t;

typedef union {
    index_slice_t slice;
    int32_t index;
} tensor_index_val_t;

typedef struct {
    index_type_t type;
    tensor_index_val_t value;
} tensor_index_t;

typedef struct {
    tensor_index_t* items;
    size_t len;
} index_tuple_t;

#endif // TENSOR_INDEX_H
