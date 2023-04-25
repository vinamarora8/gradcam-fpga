#include "util.h"
#include "conv.h"
#include "max_pool.h"
#include "avg_pool.h"
#include "linear_fc.h"

#ifdef CSIM_DEBUG
#   define WRITE_TO_FILE_ENABLED
#endif

#ifdef WRITE_TO_FILE_ENABLED
std::string root_dir = "out/";
#define WRITE_TO_FILE(var, dim0, dim1, dim2) \
    { \
        std::vector<int> dims(3); \
        dims[0] = dim0; \
        dims[1] = dim1; \
        dims[2] = dim2; \
        write_to_file(root_dir + VAR_NAME(var) + ".bin", dims, var); \
    }
#define WRITE_TO_FILE_NAME(var, name, dim0, dim1, dim2) \
    { \
        std::vector<int> dims(3); \
        dims[0] = dim0; \
        dims[1] = dim1; \
        dims[2] = dim2; \
        write_to_file(root_dir + name + ".bin", dims, var); \
    }
#else
#define WRITE_TO_FILE(var, dim0, dim1, dim2)
#define WRITE_TO_FILE_NAME(var, name, dim0, dim1, dim2)
#endif

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
    conv<3, 224, 224,
        64, 112, 112,
        7, 7, 2, 3, true, false>(conv1_out, input, conv1_weight, conv1_bias, nullptr);
    WRITE_TO_FILE(conv1_out, 64, 112, 112);


    // maxpool
    fm_t maxpool_out[64][56][56];
    maxpool2d<64, 112, 112, 
            64, 56, 56, 
            3, 3, 2, 1>(maxpool_out, conv1_out);
    WRITE_TO_FILE(maxpool_out, 64, 56, 56);

    // layer 1 
    // block 0
    fm_t l1_out0[64][56][56];
    fm_t l1_out1[64][56][56];
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, false>(l1_out0, maxpool_out, l10_c1_weight, l10_c1_bias, nullptr);
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, true>(l1_out1, l1_out0, l10_c2_weight, l10_c2_bias, maxpool_out);
    WRITE_TO_FILE_NAME(l1_out0, "l10_c1_out", 64, 56, 56);
    WRITE_TO_FILE_NAME(l1_out1, "l10_c2_out", 64, 56, 56);
    // block 1
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, false>(l1_out0, l1_out1, l11_c1_weight, l11_c1_bias, nullptr);
    conv<64, 56, 56,
        64, 56, 56,
        3, 3, 1, 1, true, true>(l1_out1, l1_out0, l11_c2_weight, l11_c2_bias, l1_out1);
    WRITE_TO_FILE_NAME(l1_out0, "l11_c1_out", 64, 56, 56);
    WRITE_TO_FILE_NAME(l1_out1, "l11_c2_out", 64, 56, 56);

    // layer 2
    // downsample
    fm_t l2_out0[128][28][28];
    fm_t l2_out1[128][28][28];
    conv<64, 56, 56,
        128, 28, 28,
        1, 1, 2, 0, true, false, false>(l2_out1, l1_out1, l2_ds_weight, l2_ds_bias, nullptr);
    WRITE_TO_FILE_NAME(l2_out1, "l2_ds_out", 128, 28, 28);
    // block 0
    conv<64, 56, 56,
        128, 28, 28,
        3, 3, 2, 1, true, false>(l2_out0, l1_out1, l20_c1_weight, l20_c1_bias, nullptr);
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, true>(l2_out1, l2_out0, l20_c2_weight, l20_c2_bias, l2_out1);
    WRITE_TO_FILE_NAME(l2_out0, "l20_c1_out", 128, 28, 28);
    WRITE_TO_FILE_NAME(l2_out1, "l20_c2_out", 128, 28, 28);
    // block 1
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, false>(l2_out0, l2_out1, l21_c1_weight, l21_c1_bias, nullptr);
    conv<128, 28, 28,
        128, 28, 28,
        3, 3, 1, 1, true, true>(l2_out1, l2_out0, l21_c2_weight, l21_c2_bias, l2_out1);
    WRITE_TO_FILE_NAME(l2_out0, "l21_c1_out", 128, 28, 28);
    WRITE_TO_FILE_NAME(l2_out1, "l21_c2_out", 128, 28, 28);

    // layer 3
    fm_t l3_out0[256][14][14];
    fm_t l3_out1[256][14][14];
    // downsample
    conv<128, 28, 28,
        256, 14, 14,
        1, 1, 2, 0, true, false, false>(l3_out1, l2_out1, l3_ds_weight, l3_ds_bias, nullptr);
    WRITE_TO_FILE_NAME(l3_out1, "l3_ds_out", 256, 14, 14);
    // block 0
    conv<128, 28, 28,
        256, 14, 14,
        3, 3, 2, 1, true, false>(l3_out0, l2_out1, l30_c1_weight, l30_c1_bias, nullptr);
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, true>(l3_out1, l3_out0, l30_c2_weight, l30_c2_bias, l3_out1);
    WRITE_TO_FILE_NAME(l3_out0, "l30_c1_out", 256, 14, 14);
    WRITE_TO_FILE_NAME(l3_out1, "l30_c2_out", 256, 14, 14);
    // block 1
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, false>(l3_out0, l3_out1, l31_c1_weight, l31_c1_bias, nullptr);
    conv<256, 14, 14,
        256, 14, 14,
        3, 3, 1, 1, true, true>(l3_out1, l3_out0, l31_c2_weight, l31_c2_bias, l3_out1);
    WRITE_TO_FILE_NAME(l3_out0, "l31_c1_out", 256, 14, 14);
    WRITE_TO_FILE_NAME(l3_out1, "l31_c2_out", 256, 14, 14);


    // layer 4
    // downsample
    fm_t l4_ds_out[512][7][7];
    conv<256, 14, 14,
        512, 7, 7,
        1, 1, 2, 0, true, false, false>(l4_ds_out, l3_out1, l4_ds_weight, l4_ds_bias, nullptr);
    WRITE_TO_FILE(l4_ds_out, 512, 7, 7);
    // block 0
    fm_t l40_c1_out[512][7][7];
    fm_t l40_c2_out[512][7][7];
    conv<256, 14, 14,
        512, 7, 7,
        3, 3, 2, 1, true, false>(l40_c1_out, l3_out1, l40_c1_weight, l40_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l40_c2_out, l40_c1_out, l40_c2_weight, l40_c2_bias, l4_ds_out);
    WRITE_TO_FILE(l40_c1_out, 512, 7, 7);
    WRITE_TO_FILE(l40_c2_out, 512, 7, 7);
    // block 1
    fm_t l41_c1_out[512][7][7];
    fm_t l41_c2_out[512][7][7];
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, false>(l41_c1_out, l40_c2_out, l41_c1_weight, l41_c1_bias, nullptr);
    conv<512, 7, 7,
        512, 7, 7,
        3, 3, 1, 1, true, true>(l41_c2_out, l41_c1_out, l41_c2_weight, l41_c2_bias, l40_c2_out);
    WRITE_TO_FILE(l41_c1_out, 512, 7, 7);
    WRITE_TO_FILE(l41_c2_out, 512, 7, 7);


    // avgpool
    fm_t avgpool_out[512];
    avg_pool<512, 7, 7>(l41_c2_out, avgpool_out);
    WRITE_TO_FILE(avgpool_out, 512, 1, 1);

    
    // fc
    linear_fc<512, 1000>(avgpool_out, output, fc_weight, fc_bias);
    WRITE_TO_FILE(output, 1000, 1, 1);
}
