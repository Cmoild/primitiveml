#ifndef DYNARRAY_HEADER_FILE
#define DYNARRAY_HEADER_FILE

#include <stddef.h>
#include <error_handling.h>
#include <container_type.h>

typedef struct dynarray {
    size_t _size;
    size_t _capacity;
    container_type_t type;
    void *_container;
    void (*print)(const struct dynarray* self);
    result_t (*get_at)(const struct dynarray* self, const size_t idx);
    result_t (*set_at)(struct dynarray* self, const size_t idx, const void* value);
    pml_err_t (*resize)(struct dynarray* self, const size_t new_capacity);
} dynarray;

dynarray dynarray_create(const void *data, const size_t len, const container_type_t type, pml_err_t *error);

#endif // DYNARRAY_HEADER_FILE
