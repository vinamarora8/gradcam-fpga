#define CSIM_DEBUG

#include "util.h"
#include "conv.h"
#include "max_pool.h"
#include "avg_pool.h"
#include "linear_fc.h"

void resnet18(
        fm_t input[3][224][224],
        fm_t output[1000],
        wt_t conv1_weight[64][3][7][7],
        wt_t conv1_bias[64],
        // layer 1
        wt_t l1_c1_weight[64][64][3][3],
        wt_t l1_c1_bias[64],
        wt_t l1_c2_weight[64][64][3][3],
        wt_t l1_c2_bias[64],
        // layer 2
        wt_t l2_c1_weight[128][64][3][3],
        wt_t l2_c1_bias[128],
        wt_t l2_c2_weight[128][128][3][3],
        wt_t l2_c2_bias[128],
        wt_t l2_ds_weight[128][64][1][1],
        wt_t l2_ds_bias[128],
        // layer 3
        wt_t l3_c1_weight[256][128][3][3],
        wt_t l3_c1_bias[256],
        wt_t l3_c2_weight[256][256][3][3],
        wt_t l3_c2_bias[256],
        wt_t l3_ds_weight[256][128][1][1],
        wt_t l3_ds_bias[256],
        // layer 4
        wt_t l4_c1_weight[512][256][3][3],
        wt_t l4_c1_bias[512],
        wt_t l4_c2_weight[512][512][3][3],
        wt_t l4_c2_bias[512],
        wt_t l4_ds_weight[512][256][1][1],
        wt_t l4_ds_bias[512],
        // fc
        wt_t fc_weight[1000][512],
        wt_t fc_bias[1000],

        // activations
        fm_t conv1_out[64][112][112],
        fm_t maxpool_out[64][56][56],
        fm_t l1_c1_out[64][56][56],
        fm_t l1_c2_out[64][56][56],
        fm_t l2_ds_out[128][28][28],
        fm_t l2_c1_out[128][28][28],
        fm_t l2_c2_out[128][28][28],
        fm_t l3_ds_out[256][14][14],
        fm_t l3_c1_out[256][14][14],
        fm_t l3_c2_out[256][14][14],
        fm_t l4_ds_out[512][7][7],
        fm_t l4_c1_out[512][7][7],
        fm_t l4_c2_out[512][7][7],
        fm_t avgpool_out[512]
        )
{
    // conv1
    conv<3, 224, 224,
        64, 112, 112,
        7, 7, 2, 3, true, false>(conv1_out, input, conv1_weight, conv1_bias, nullptr);

    // maxpool
    maxpool2d<64, 112, 112, 
            64, 56, 56, 
            3, 3, 2, 1>(maxpool_out, conv1_out);

    // layer 1
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, false>(l1_c1_out, maxpool_out, l1_c1_weight, l1_c1_bias, nullptr);
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, true>(l1_c2_out, l1_c1_out, l1_c2_weight, l1_c2_bias, maxpool_out);


    // layer 2
    conv<64, 56, 56,
        128, 28, 28,
        1, 1, 2, 0, true, false>(l2_ds_out, l1_c2_out, l2_ds_weight, l2_ds_bias, nullptr);

    conv<64, 56, 56,
        128, 28, 28,
        3, 3, 1, 1, true, false>(l2_c1_out, l1_c2_out, l2_c1_weight, l2_c1_bias, nullptr);
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, true>(l2_c2_out, l2_c1_out, l2_c2_weight, l2_c2_bias, l2_ds_out);

    // layer 3
    conv<128, 28, 28,
        256, 14, 14,
        1, 1, 2, 0, true, false>(l3_ds_out, l2_c2_out, l3_ds_weight, l3_ds_bias, nullptr);

    conv<128, 28, 28,
        256, 14, 14,
        3, 3, 2, 1, true, false>(l3_c1_out, l2_c2_out, l3_c1_weight, l3_c1_bias, nullptr);
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, true>(l3_c2_out, l3_c1_out, l3_c2_weight, l3_c2_bias, l3_ds_out);

    // layer 4
    conv<256, 14, 14,
        512, 7, 7,
        1, 1, 2, 0, true, false>(l4_ds_out, l3_c2_out, l4_ds_weight, l4_ds_bias, nullptr);
    
    conv<256, 14, 14,
        512, 7, 7,
        3, 3, 1, 1, true, false>(l4_c1_out, l3_c2_out, l4_c1_weight, l4_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l4_c2_out, l4_c1_out, l4_c2_weight, l4_c2_bias, l4_ds_out);


    // avgpool
    avg_pool<512, 7, 7>(l4_c2_out, avgpool_out);

    
    // fc
    linear_fc<512, 1000, true>(avgpool_out, output, fc_weight, fc_bias);
}
