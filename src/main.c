#include <stdio.h>
#include <dynarray.h>
#include <tensor.h>
#include <stdlib.h>
#include <module.h>
#include <linear.h>
#include <functional.h>
#include <tensor_index.h>
#include <embedding.h>
#include <math.h>
#include <gpt_modules.h>


int main(){
    pml_err_t err = PML_OK;

    float qkv_raw[] = { 0.3552,  0.3065,  0.4564, -0.1883,
        0.3953,  0.4493, -0.4293, -0.3116,
        -0.0231,  0.3418,  0.3982,  0.0919,
        0.3515,  0.3539,  0.0534, -0.2218,
        -0.2413, -0.0310, -0.0917, -0.4214,
        0.2419,  0.1282,  0.4402,  0.0096,
        -0.3517,  0.1292,  0.0345,  0.0897,
        0.0844, -0.2975, -0.1901, -0.2388,
        -0.3261,  0.2642,  0.2528, -0.4918,
        0.1302,  0.4182,  0.0712, -0.1400,
        -0.1607,  0.3022, -0.1465, -0.0785,
        0.0283,  0.3751,  0.3646,  0.3555};
    
    float out_raw[] = {0.4213,  0.2141, -0.4867,  0.3417,
        -0.4254,  0.2888,  0.4202, -0.0359,
        0.3299,  0.1581, -0.2581, -0.3107,
        0.4140,  0.0648, -0.3310,  0.4397};

    float inp_raw[] = { 1.5545,  0.5008, -0.5759,  0.7472,
         0.8077, -1.8945, -2.1331,  0.4553,
         -0.1906, -0.3576,  0.2012, -0.8386,
         1.4127, -0.0786, -0.8873, -0.1805,

        -0.1626, -0.3935,  0.3706,  0.2927,
         -1.1251, -0.2035, -0.3665,  0.4556,
         -0.1743, -0.3018,  0.8250, -0.0689,
          1.7742, -1.2774,  0.9305,  2.0606};

    tensor* mask = tensor_create(
        (float[]){0, -INFINITY, -INFINITY, -INFINITY, 0, 0, -INFINITY, -INFINITY, 0, 0, 0, -INFINITY, 0, 0, 0, 0},
        16, TYPE_FLOAT, 4, dynarray_create((int32_t[]){1, 1, 4, 4}, 4, TYPE_INT32, &err), &err
    );
    tensor* qkv_proj_weight = tensor_create(
        qkv_raw, 48, TYPE_FLOAT, 2, dynarray_create((int[]){12, 4}, 2, TYPE_INT32, &err), &err
    );
    tensor* qkv_proj_bias = tensor_create(
        (float[]){-0.1176, -0.3661, -0.4420, -0.2758, -0.1378,  0.0583, -0.0369, -0.0758, -0.2689,  0.1638,  0.4399,  0.2850},
        12, TYPE_FLOAT, 1, dynarray_create((int[]){12}, 1, TYPE_INT32, &err), &err
    );
    linear_module* qkv_proj = linear_module_create(qkv_proj_weight, qkv_proj_bias, &err);
    tensor* out_proj_weight = tensor_create(
        out_raw, 16, TYPE_FLOAT, 2, dynarray_create((int[]){4, 4}, 2, TYPE_INT32, &err), &err
    );
    tensor* out_proj_bias = tensor_create(
        (float[]){-0.2574, -0.3778, -0.3330,  0.1331},
        4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    linear_module* out_proj = linear_module_create(out_proj_weight, out_proj_bias, &err);

    causal_self_attention* mha = causal_self_attention_create(qkv_proj, out_proj, mask, 2, 4, &err);

    tensor* inp = tensor_create(
        inp_raw,
        32, TYPE_FLOAT, 3, dynarray_create((int[]){2, 4, 4}, 3, TYPE_INT32, &err), &err
    );

    tensor* out = mha->module_base.forward(mha, inp);

    out->print(out);
    
    return 0;
}
