#ifndef FUNCTIONAL_HEADER_FILE
#define FUNCTIONAL_HEADER_FILE

#include <tensor.h>
#include <error_handling.h>

tensor* relu(const tensor* input, pml_err_t* err);

tensor* sigmoid(const tensor* input, pml_err_t* err);

tensor* softmax(const tensor* input, int32_t dim, pml_err_t* err);

#endif // FUNCTIONAL_HEADER_FILE