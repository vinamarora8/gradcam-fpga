#include "util.h"
#include "conv.h"
#include "max_pool.h"
#include "avg_pool.h"
#include "linear_fc.h"
#include "tiled_conv/tiled_conv.cpp"
#include "sim_util.h"
#include "residual.cpp"

void resnet18(
        fm_t input[3][224][224],
        fm_t output[1000],
        wt_t conv1_weight[64][3][7][7],
        wt_t conv1_bias[64],
        // layer 1
        wt_t l10_c1_weight[64][64][3][3],
        wt_t l10_c1_bias[64],
        wt_t l10_c2_weight[64][64][3][3],
        wt_t l10_c2_bias[64],
        wt_t l11_c1_weight[64][64][3][3],
        wt_t l11_c1_bias[64],
        wt_t l11_c2_weight[64][64][3][3],
        wt_t l11_c2_bias[64],
        // layer 2
        wt_t l2_ds_weight[128][64][1][1],
        wt_t l2_ds_bias[128],
        wt_t l20_c1_weight[128][64][3][3],
        wt_t l20_c1_bias[128],
        wt_t l20_c2_weight[128][128][3][3],
        wt_t l20_c2_bias[128],
        wt_t l21_c1_weight[128][128][3][3],
        wt_t l21_c1_bias[128],
        wt_t l21_c2_weight[128][128][3][3],
        wt_t l21_c2_bias[128],
        // layer 3
        wt_t l3_ds_weight[256][128][1][1],
        wt_t l3_ds_bias[256],
        wt_t l30_c1_weight[256][128][3][3],
        wt_t l30_c1_bias[256],
        wt_t l30_c2_weight[256][256][3][3],
        wt_t l30_c2_bias[256],
        wt_t l31_c1_weight[256][256][3][3],
        wt_t l31_c1_bias[256],
        wt_t l31_c2_weight[256][256][3][3],
        wt_t l31_c2_bias[256],
        // layer 4
        wt_t l4_ds_weight[512][256][1][1],
        wt_t l4_ds_bias[512],
        wt_t l40_c1_weight[512][256][3][3],
        wt_t l40_c1_bias[512],
        wt_t l40_c2_weight[512][512][3][3],
        wt_t l40_c2_bias[512],
        wt_t l41_c1_weight[512][512][3][3],
        wt_t l41_c1_bias[512],
        wt_t l41_c2_weight[512][512][3][3],
        wt_t l41_c2_bias[512],
        // fc
        wt_t fc_weight[1000][512],
        wt_t fc_bias[1000]
        )
{
    

    WRITE_TO_FILE(input, 3, 224, 224);

    // conv1
    fm_t conv1_out[64][112][112];
    tiled_conv
        <64, 3, 7, 7, 2, 3,
        224, 224, 64, 32, 32>
        (conv1_out, input, conv1_weight, conv1_bias, true);
    WRITE_TO_FILE(conv1_out, 64, 112, 112);


    // maxpool
    fm_t maxpool_out[64][56][56];
    maxpool2d<64, 112, 112, 
            64, 56, 56, 
            3, 3, 2, 1>(maxpool_out, conv1_out);
    WRITE_TO_FILE(maxpool_out, 64, 56, 56);

    fm_t maxpool_out_copy[64][56][56];
    for (int i = 0; i < 64*56*56; i++)
    {
        maxpool_out_copy[0][0][i] = maxpool_out[0][0][i];
    }

    // layer 1 
    // block 0
    fm_t l1_out0[64][56][56];
    // Testing tiled_conv
    tiled_conv
        <64, 64, 3, 3, 1, 1,
        56, 56, 64, 7, 7>
    (l1_out0, maxpool_out, l10_c1_weight, l10_c1_bias, true);
    tiled_conv
        <64, 64, 3, 3, 1, 1,
        56, 56, 64, 7, 7>
    (maxpool_out, l1_out0, l10_c2_weight, l10_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l1_out0, "l10_c1_out", 64, 56, 56);
    WRITE_TO_FILE_NAME(maxpool_out, "l10_c2_out", 64, 56, 56);
    // block 1
    tiled_conv
        <64, 64, 3, 3, 1, 1,
        56, 56, 64, 7, 7>
    (l1_out0, maxpool_out, l11_c1_weight, l11_c1_bias, true);
    tiled_conv
        <64, 64, 3, 3, 1, 1,
        56, 56, 64, 7, 7>
    (maxpool_out, l1_out0, l11_c2_weight, l11_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l1_out0, "l11_c1_out", 64, 56, 56);
    WRITE_TO_FILE_NAME(maxpool_out, "l11_c2_out", 64, 56, 56);

    // layer 2
    // downsample
    fm_t l2_out0[128][28][28];
    fm_t l2_out1[128][28][28];
    tiled_conv
        <128, 64, 1, 1, 2, 0,
        56, 56, 64, 14, 14>
        (l2_out1, maxpool_out, l2_ds_weight, l2_ds_bias, false);
    WRITE_TO_FILE_NAME(l2_out1, "l2_ds_out", 128, 28, 28);
    // block 0
    tiled_conv
        <128, 64, 3, 3, 2, 1,
        56, 56, 64, 14, 14>
        (l2_out0, maxpool_out, l20_c1_weight, l20_c1_bias, true);
    tiled_conv
        <128, 128, 3, 3, 1, 1,
        28, 28, 128, 7, 7>
        (l2_out1, l2_out0, l20_c2_weight, l20_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l2_out0, "l20_c1_out", 128, 28, 28);
    WRITE_TO_FILE_NAME(l2_out1, "l20_c2_out", 128, 28, 28);
    // block 1
    tiled_conv
        <128, 128, 3, 3, 1, 1,
        28, 28, 128, 7, 7>
        (l2_out0, l2_out1, l21_c1_weight, l21_c1_bias, true);
    tiled_conv
        <128, 128, 3, 3, 1, 1,
        28, 28, 128, 7, 7>
        (l2_out1, l2_out0, l21_c2_weight, l21_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l2_out0, "l21_c1_out", 128, 28, 28);
    WRITE_TO_FILE_NAME(l2_out1, "l21_c2_out", 128, 28, 28);

    // layer 3
    fm_t l3_out0[256][14][14];
    fm_t l3_out1[256][14][14];
    // downsample
    tiled_conv
        <256, 128, 1, 1, 2, 0,
        28, 28, 64, 14, 14>
        (l3_out1, l2_out1, l3_ds_weight, l3_ds_bias, false);
    WRITE_TO_FILE_NAME(l3_out1, "l3_ds_out", 256, 14, 14);
    // block 0
    tiled_conv
        <256, 128, 3, 3, 2, 1,
        28, 28, 64, 14, 14>
        (l3_out0, l2_out1, l30_c1_weight, l30_c1_bias, true);
    tiled_conv
        <256, 256, 3, 3, 1, 1,
        14, 14, 64, 7, 7>
        (l3_out1, l3_out0, l30_c2_weight, l30_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l3_out0, "l30_c1_out", 256, 14, 14);
    WRITE_TO_FILE_NAME(l3_out1, "l30_c2_out", 256, 14, 14);
    // block 1
    tiled_conv
        <256, 256, 3, 3, 1, 1,
        14, 14, 64, 7, 7>
        (l3_out0, l3_out1, l31_c1_weight, l31_c1_bias, true);
    tiled_conv
        <256, 256, 3, 3, 1, 1,
        14, 14, 64, 7, 7>
        (l3_out1, l3_out0, l31_c2_weight, l31_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l3_out0, "l31_c1_out", 256, 14, 14);
    WRITE_TO_FILE_NAME(l3_out1, "l31_c2_out", 256, 14, 14);


    // layer 4
    fm_t l4_out0[512][7][7];
    fm_t l4_out1[512][7][7];
    // downsample
    conv<256, 14, 14,
        512, 7, 7,
        1, 1, 2, 0, true, false, false>(l4_out1, l3_out1, l4_ds_weight, l4_ds_bias, nullptr);
    WRITE_TO_FILE_NAME(l4_out1, "l4_ds_out", 512, 7, 7);
    // block 0
    conv<256, 14, 14,
        512, 7, 7,
        3, 3, 2, 1, true, false>(l4_out0, l3_out1, l40_c1_weight, l40_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l4_out1, l4_out0, l40_c2_weight, l40_c2_bias, l4_out1);
    WRITE_TO_FILE_NAME(l4_out0, "l40_c1_out", 512, 7, 7);
    WRITE_TO_FILE_NAME(l4_out1, "l40_c2_out", 512, 7, 7);
    // block 1
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, false>(l4_out0, l4_out1, l41_c1_weight, l41_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l4_out1, l4_out0, l41_c2_weight, l41_c2_bias, l4_out1);
    WRITE_TO_FILE_NAME(l4_out0, "l41_c1_out", 512, 7, 7);
    WRITE_TO_FILE_NAME(l4_out1, "l41_c2_out", 512, 7, 7);


    // avgpool
    fm_t avgpool_out[512];
    avg_pool<512, 7, 7>(l4_out1, avgpool_out);
    WRITE_TO_FILE(avgpool_out, 512, 1, 1);

    
    // fc
    linear_fc<512, 1000>(avgpool_out, output, fc_weight, fc_bias);
    WRITE_TO_FILE(output, 1000, 1, 1);
}
