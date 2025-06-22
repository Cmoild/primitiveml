#ifndef MODULE_HEADER_FILE
#define MODULE_HEADER_FILE

#include <tensor.h>

typedef struct module_iface {

    tensor* (*forward)(const void* self, const tensor* input);
    void (*destroy)(void* self);
    void (*from_file)(const void* self, const char* path);
    void (*print)(const void* self);

} module_iface;

#endif // MODULE_HEADER_FILE