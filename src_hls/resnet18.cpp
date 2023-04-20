#define CSIM_DEBUG

#include "util.h"
#include "conv.h"
#include "max_pool.h"
#include "avg_pool.h"
#include "linear_fc.h"

//#define WRITE_TO_FILE_ENABLED

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
    
    std::string root_dir = "out/";

    // conv1
    fm_t conv1_out[64][112][112];
    conv<3, 224, 224,
        64, 112, 112,
        7, 7, 2, 3, true, false>(conv1_out, input, conv1_weight, conv1_bias, nullptr);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(conv1_out), {64,112,112}, conv1_out);
    #endif


    // maxpool
    fm_t maxpool_out[64][56][56];
    maxpool2d<64, 112, 112, 
            64, 56, 56, 
            3, 3, 2, 1>(maxpool_out, conv1_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(maxpool_out), {64,56,56}, maxpool_out);
    #endif

    // layer 1 
    // block 0
    fm_t l10_c1_out[64][56][56];
    fm_t l10_c2_out[64][56][56];
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, false>(l10_c1_out, maxpool_out, l10_c1_weight, l10_c1_bias, nullptr);
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, true>(l10_c2_out, l10_c1_out, l10_c2_weight, l10_c2_bias, maxpool_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l10_c1_out), {64,56,56}, l10_c1_out);
        write_to_file(root_dir + VAR_NAME(l10_c2_out), {64,56,56}, l10_c2_out);
    #endif
    // block 1
    fm_t l11_c1_out[64][56][56];
    fm_t l11_c2_out[64][56][56];
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, false>(l11_c1_out, l10_c1_out, l11_c1_weight, l11_c1_bias, nullptr);
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, true>(l11_c2_out, l11_c1_out, l11_c2_weight, l11_c2_bias, l10_c2_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l11_c1_out), {64,56,56}, l11_c1_out);
        write_to_file(root_dir + VAR_NAME(l11_c2_out), {64,56,56}, l11_c2_out);
    #endif


    // layer 2
    // downsample
    fm_t l2_ds_out[128][28][28];
    conv<64, 56, 56,
        128, 28, 28,
        1, 1, 2, 0, true, false>(l2_ds_out, l11_c2_out, l2_ds_weight, l2_ds_bias, nullptr);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l2_ds_out), {128,28,28}, l2_ds_out);
    #endif
    // block 0
    fm_t l20_c1_out[128][28][28];
    fm_t l20_c2_out[128][28][28];
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l20_c1_out), {128,28,28}, l20_c1_out);
        write_to_file(root_dir + VAR_NAME(l20_c2_out), {128,28,28}, l20_c2_out);
    #endif
    conv<64, 56, 56,
        128, 28, 28,
        3, 3, 2, 1, true, false>(l20_c1_out, l11_c2_out, l20_c1_weight, l20_c1_bias, nullptr);
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, true>(l20_c2_out, l20_c1_out, l20_c2_weight, l20_c2_bias, l2_ds_out);
    // block 1
    fm_t l21_c1_out[128][28][28];
    fm_t l21_c2_out[128][28][28];
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, false>(l21_c1_out, l20_c2_out, l21_c1_weight, l21_c1_bias, nullptr);
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, true>(l21_c2_out, l21_c1_out, l21_c2_weight, l21_c2_bias, l20_c2_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l21_c1_out), {128,28,28}, l21_c1_out);
        write_to_file(root_dir + VAR_NAME(l21_c2_out), {128,28,28}, l21_c2_out);
    #endif

    // layer 3
    // downsample
    fm_t l3_ds_out[256][14][14];
    conv<128, 28, 28,
        256, 14, 14,
        1, 1, 2, 0, true, false>(l3_ds_out, l21_c2_out, l3_ds_weight, l3_ds_bias, nullptr);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l3_ds_out), {256,14,14}, l3_ds_out);
    #endif
    // block 0
    fm_t l30_c1_out[256][14][14];
    fm_t l30_c2_out[256][14][14];
    conv<128, 28, 28,
        256, 14, 14,
        3, 3, 2, 1, true, false>(l30_c1_out, l21_c2_out, l30_c1_weight, l30_c1_bias, nullptr);
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, true>(l30_c2_out, l30_c1_out, l30_c2_weight, l30_c2_bias, l3_ds_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l30_c1_out), {256,14,14}, l30_c1_out);
        write_to_file(root_dir + VAR_NAME(l30_c2_out), {256,14,14}, l30_c2_out);
    #endif
    // block 1
    fm_t l31_c1_out[256][14][14];
    fm_t l31_c2_out[256][14][14];
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, false>(l31_c1_out, l30_c2_out, l31_c1_weight, l31_c1_bias, nullptr);
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, true>(l31_c2_out, l31_c1_out, l31_c2_weight, l31_c2_bias, l30_c2_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l31_c1_out), {256,14,14}, l31_c1_out);
        write_to_file(root_dir + VAR_NAME(l31_c2_out), {256,14,14}, l31_c2_out);
    #endif


    // layer 4
    // downsample
    fm_t l4_ds_out[512][7][7];
    conv<256, 14, 14,
        512, 7, 7,
        1, 1, 2, 0, true, false>(l4_ds_out, l31_c2_out, l4_ds_weight, l4_ds_bias, nullptr);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l4_ds_out), {512,7,7}, l4_ds_out);
    #endif
    // block 0
    fm_t l40_c1_out[512][7][7];
    fm_t l40_c2_out[512][7][7];
    conv<256, 14, 14,
        512, 7, 7,
        3, 3, 2, 1, true, false>(l40_c1_out, l31_c2_out, l40_c1_weight, l40_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l40_c2_out, l40_c1_out, l40_c2_weight, l40_c2_bias, l4_ds_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l40_c1_out), {512,7,7}, l40_c1_out);
        write_to_file(root_dir + VAR_NAME(l40_c2_out), {512,7,7}, l40_c2_out);
    #endif
    // block 1
    fm_t l41_c1_out[512][7][7];
    fm_t l41_c2_out[512][7][7];
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, false>(l41_c1_out, l40_c2_out, l41_c1_weight, l41_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l41_c2_out, l41_c1_out, l41_c2_weight, l41_c2_bias, l40_c2_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(l41_c1_out), {512,7,7}, l41_c1_out);
        write_to_file(root_dir + VAR_NAME(l41_c2_out), {512,7,7}, l41_c2_out);
    #endif


    // avgpool
    fm_t avgpool_out[512];
    avg_pool<512, 7, 7>(l41_c2_out, avgpool_out);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(avgpool_out), {512}, avgpool_out);
    #endif

    
    // fc
    linear_fc<512, 1000, true>(avgpool_out, output, fc_weight, fc_bias);
    #ifdef WRITE_TO_FILE_ENABLED
        write_to_file(root_dir + VAR_NAME(output), {1000}, output);
    #endif
}