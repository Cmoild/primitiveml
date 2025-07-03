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
#include <layernorm.h>


int main(){
    pml_err_t err = PML_OK;

    float qkv_raw[] = { -0.0878, -0.4265, -0.3505,  0.3597,
         0.3823,  0.3457, -0.1671,  0.3108,
        -0.2894, -0.4214, -0.0571,  0.1501,
         0.1977,  0.2233, -0.3944, -0.2411,
         0.4576, -0.1774, -0.3601,  0.4687,
         0.2580,  0.1630, -0.2478, -0.1818,
         0.4050, -0.4734,  0.2719, -0.2527,
        -0.4110, -0.1493,  0.2241,  0.3029,
        -0.3172, -0.4394,  0.1445, -0.4184,
         0.1944, -0.2957, -0.0782, -0.1530,
         0.1637,  0.1336,  0.1530, -0.1871,
         0.1022, -0.4799,  0.0123,  0.4877};
    
    float out_raw[] = {-0.0356, -0.0921, -0.1591,  0.3990,
         0.1694,  0.1520,  0.0902,  0.1243,
        -0.3209,  0.4443,  0.3794, -0.0103,
         0.0766, -0.4241,  0.2006,  0.1896};

    float ffn_in_raw[] = {
        -0.4185,  0.0086, -0.1012,  0.3990,
        -0.0426, -0.3150,  0.2708,  0.4022,
        -0.2615, -0.4997, -0.2001,  0.0022,
         0.4716, -0.1254, -0.1994,  0.4800,
        -0.0165, -0.2025, -0.0788, -0.2428,
         0.2750,  0.3524, -0.3218, -0.1378,
         0.0210,  0.3411, -0.3459,  0.1607,
        -0.2337, -0.3507,  0.1284, -0.0073,
         0.2477, -0.1321, -0.1550,  0.0553,
         0.4768, -0.4789, -0.4746, -0.2550,
        -0.0862,  0.3918, -0.0342, -0.2020,
         0.3362, -0.2335,  0.4472,  0.3023,
         0.1733, -0.3132,  0.1478,  0.3518,
         0.4275,  0.1094,  0.4205, -0.4440,
        -0.2255, -0.4458,  0.2760, -0.0807,
        -0.3618,  0.3078,  0.3172,  0.1798
    };

    float ffn_out_raw[] = {
         0.1879, -0.2004,  0.0953, -0.1690, -0.0432, -0.1536,  0.2481,  0.0822,
          0.0900, -0.1814, -0.1860, -0.0646, -0.0794, -0.0870, -0.2492, -0.0518,
        -0.1729, -0.1231,  0.1260,  0.1565, -0.1478, -0.0282,  0.1398,  0.2282,
          0.0382,  0.0155, -0.1143,  0.1672,  0.2245, -0.0840,  0.1418, -0.0433,
         0.0240,  0.1384, -0.0241, -0.1637,  0.0745,  0.0684, -0.2021,  0.0466,
          0.2062,  0.2348, -0.2108,  0.1144, -0.1796,  0.0235,  0.0245, -0.0662,
        -0.0361,  0.1584, -0.0062,  0.2195, -0.0978,  0.0100, -0.2251, -0.1219,
         -0.1154, -0.1685,  0.0815, -0.1440,  0.2125,  0.2140, -0.1560,  0.1669
    };

    float inp_raw[] = {
        0.5814, -1.3704, -1.2674,  0.8584,
         -1.4031,  0.7557,  0.1565,  1.1779,
         -1.0863, -0.1881,  0.4287,  0.1028
    };

    tensor* mask = tensor_create(
        (float[]){0, -INFINITY, -INFINITY, -INFINITY, 0, 0, -INFINITY, -INFINITY, 0, 0, 0, -INFINITY, 0, 0, 0, 0},
        16, TYPE_FLOAT, 4, dynarray_create((int32_t[]){1, 1, 4, 4}, 4, TYPE_INT32, &err), &err
    );
    tensor* qkv_proj_weight = tensor_create(
        qkv_raw, 48, TYPE_FLOAT, 2, dynarray_create((int[]){12, 4}, 2, TYPE_INT32, &err), &err
    );
    tensor* qkv_proj_bias = tensor_create(
        (float[]){0.3668, -0.0126, -0.4240,  0.0527,  0.3913,  0.0207, -0.3088, -0.1625,-0.1306, -0.3959,  0.1611,  0.0813},
        12, TYPE_FLOAT, 1, dynarray_create((int[]){12}, 1, TYPE_INT32, &err), &err
    );
    linear_module* qkv_proj = linear_module_create(qkv_proj_weight, qkv_proj_bias, &err);

    tensor* out_proj_weight = tensor_create(
        out_raw, 16, TYPE_FLOAT, 2, dynarray_create((int[]){4, 4}, 2, TYPE_INT32, &err), &err
    );
    tensor* out_proj_bias = tensor_create(
        (float[]){-0.1099, -0.4050, -0.2568,  0.0227},
        4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    linear_module* out_proj = linear_module_create(out_proj_weight, out_proj_bias, &err);

    causal_self_attention* mha = causal_self_attention_create(qkv_proj, out_proj, mask, 2, 4, &err);

    tensor* inp = tensor_create(
        inp_raw,
        12, TYPE_FLOAT, 3, dynarray_create((int[]){1, 3, 4}, 3, TYPE_INT32, &err), &err
    );

    tensor* ln1_weight = tensor_create(
        (float[]){1,1,1,1}, 4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    tensor* ln1_bias = tensor_create(
        (float[]){0,0,0,0}, 4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    tensor* ln2_weight = tensor_create(
        (float[]){1,1,1,1}, 4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    tensor* ln2_bias = tensor_create(
        (float[]){0,0,0,0}, 4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    layernorm* ln1 = layernorm_create(ln1_weight, ln1_bias, 1e-5, &err);
    layernorm* ln2 = layernorm_create(ln2_weight, ln2_bias, 1e-5, &err);

    tensor* ffn_in_weight = tensor_create(
        ffn_in_raw, 64, TYPE_FLOAT, 2, dynarray_create((int[]){16, 4}, 2, TYPE_INT32, &err), &err
    );
    tensor* ffn_in_bias = tensor_create(
        (float[]){-0.4907,  0.3957,  0.1867,  0.2088,  0.1182, -0.4857, -0.1285, -0.1062,
        -0.3871, -0.2513,  0.4071, -0.0461, -0.1526,  0.2267,  0.0039,  0.2442},
        16, TYPE_FLOAT, 1, dynarray_create((int[]){16}, 1, TYPE_INT32, &err), &err
    );
    linear_module* ffn_in = linear_module_create(ffn_in_weight, ffn_in_bias, &err);

    tensor* ffn_out_weight = tensor_create(
        ffn_out_raw, 64, TYPE_FLOAT, 2, dynarray_create((int[]){4, 16}, 2, TYPE_INT32, &err), &err
    );
    tensor* ffn_out_bias = tensor_create(
        (float[]){-0.0652,  0.1696,  0.0231,  0.1407},
        4, TYPE_FLOAT, 1, dynarray_create((int[]){4}, 1, TYPE_INT32, &err), &err
    );
    linear_module* ffn_out = linear_module_create(ffn_out_weight, ffn_out_bias, &err);

    feed_forward* ffn = feed_forward_create(ffn_in, ffn_out, &err);

    gpt_block* block = gpt_block_create(ln1, ln2, mha, ffn);

    tensor* out = block->module_base.forward(block, inp);

    out->print(out);
    
    return 0;
}
