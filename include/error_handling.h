#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

typedef enum pml_err_t {
    PML_OK = 0,
    PML_WRONG_TYPE = 1,
    PML_OUT_OF_MEMORY = 2,
    PML_OUT_OF_BOUNDS = 3,
} pml_err_t;

#endif // ERROR_HANDLING_H
