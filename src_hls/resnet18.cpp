#include "util.h"
#include "conv.h"
#include "max_pool.h"
#include "avg_pool.h"
#include "linear_fc.h"
#include "tiled_conv/tiled_conv.cpp"
#include "sim_util.h"
#include "residual.cpp"

#define INP_SIDE 224
#define INP_DEPTH 3

#define CONV1_SIDE (INP_SIDE/2)
#define CONV1_DEPTH 64
#define CONV1_SIZE (CONV1_DEPTH * CONV1_SIDE * CONV1_SIDE)
#define MAXPOOL_SIDE (CONV1_SIDE/2)
#define MAXPOOL_DEPTH (CONV1_DEPTH)
#define MAXPOOL_SIZE (MAXPOOL_DEPTH * MAXPOOL_SIDE * MAXPOOL_SIDE)

#define L1_SIDE (CONV1_SIDE/2)
#define L1_DEPTH (CONV1_DEPTH)
#define L1_SIZE (L1_DEPTH * L1_SIDE * L1_SIDE)

#define L2_SIDE (L1_SIDE/2)
#define L2_DEPTH (L1_DEPTH*2)
#define L2_SIZE (L2_DEPTH * L2_SIDE * L2_SIDE)

#define L3_SIDE (L2_SIDE/2)
#define L3_DEPTH (L2_DEPTH*2)
#define L3_SIZE (L3_DEPTH * L3_SIDE * L3_SIDE)

#define L4_SIDE (L3_SIDE/2)
#define L4_DEPTH (L3_DEPTH*2)
#define L4_SIZE (L4_DEPTH * L4_SIDE * L4_SIDE)

#define AVG_POOL_SIZE L4_DEPTH
#define OUTPUT_SIZE 1000

// FM_DRAM offsets
#define CONV1_FM_OFFSET 0
#define MAXPOOL_FM_OFFSET (CONV1_FM_OFFSET + CONV1_SIZE)
#define L1_FM_OFFSET (MAXPOOL_FM_OFFSET + MAXPOOL_SIZE)
#define L2_FM_OFFSET (L1_FM_OFFSET + 2*L1_SIZE)
#define L3_FM_OFFSET (L2_FM_OFFSET + 2*L2_SIZE)
#define L4_FM_OFFSET (L3_FM_OFFSET + 2*L3_SIZE)
#define AVG_POOL_OFFSET (L4_FM_OFFSET + 2*L4_SIZE)
#define OUTPUT_OFFSET (AVG_POOL_OFFSET + AVG_POOL_SIZE)

#define FM_DRAM_SIZE (OUTPUT_OFFSET + OUTPUT_SIZE)

void resnet18(
        fm_t input[3][224][224],
        fm_t output[1000],
        fm_t fm_dram[],
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

    fm_t *conv1_out = fm_dram;
    fm_t *maxpool_out = fm_dram + MAXPOOL_FM_OFFSET;
    fm_t *l1_out0 = fm_dram + L1_FM_OFFSET;
    fm_t *l1_out1 = maxpool_out;
    fm_t *l2_out0 = fm_dram + L2_FM_OFFSET;
    fm_t *l2_out1 = l2_out0 + L2_SIZE;
    fm_t *l3_out0 = fm_dram + L3_FM_OFFSET;
    fm_t *l3_out1 = l3_out0 + L3_SIZE;
    fm_t *l4_out0 = fm_dram + L4_FM_OFFSET;
    fm_t *l4_out1 = l4_out0 + L4_SIZE;
    fm_t *avgpool_out = fm_dram + AVG_POOL_OFFSET;

    // conv1
    tiled_conv
        <64, 3, 7, 7, 2, 3,
        224, 224, 64, 32, 32>
        (conv1_out, (fm_t *) input, conv1_weight, conv1_bias, true);
    WRITE_TO_FILE(conv1_out, 64, 112, 112);


    // maxpool
    maxpool2d<64, 112, 112, 
            64, 56, 56, 
            3, 3, 2, 1>((fm_t (*)[56][56])maxpool_out, (fm_t (*)[112][112])conv1_out);
    WRITE_TO_FILE(maxpool_out, 64, 56, 56);

    // layer 1 
    // block 0
    tiled_conv
        <L1_DEPTH, MAXPOOL_DEPTH, 3, 3, 1, 1,
        L1_SIDE, L1_SIDE, 64, 7, 7>
    (l1_out0, l1_out1, l10_c1_weight, l10_c1_bias, true);
    tiled_conv
        <L1_DEPTH, L1_DEPTH, 3, 3, 1, 1,
        L1_SIDE, L1_SIDE, 64, 7, 7>
    (l1_out1, l1_out0, l10_c2_weight, l10_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l1_out0, "l10_c1_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    WRITE_TO_FILE_NAME(l1_out1, "l10_c2_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    // block 1
    tiled_conv
        <L1_DEPTH, L1_DEPTH, 3, 3, 1, 1,
        L1_SIDE, L1_SIDE, 64, 7, 7>
    (l1_out0, l1_out1, l11_c1_weight, l11_c1_bias, true);
    tiled_conv
        <L1_DEPTH, L1_DEPTH, 3, 3, 1, 1,
        L1_SIDE, L1_SIDE, 64, 7, 7>
    (l1_out1, l1_out0, l11_c2_weight, l11_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l1_out0, "l11_c1_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    WRITE_TO_FILE_NAME(l1_out1, "l11_c2_out", L1_DEPTH, L1_SIDE, L1_SIDE);

    // layer 2
    // downsample
    tiled_conv
        <L2_DEPTH, L1_DEPTH, 1, 1, 2, 0,
        L1_SIDE, L1_SIDE, 64, 14, 14>
        (l2_out1, maxpool_out, l2_ds_weight, l2_ds_bias, false);
    WRITE_TO_FILE_NAME(l2_out1, "l2_ds_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    // block 0
    tiled_conv
        <L2_DEPTH, L1_DEPTH, 3, 3, 2, 1,
        L1_SIDE, L1_SIDE, 64, 14, 14>
        (l2_out0, maxpool_out, l20_c1_weight, l20_c1_bias, true);
    tiled_conv
        <L2_DEPTH, L2_DEPTH, 3, 3, 1, 1,
        L2_SIDE, L2_SIDE, 64, 7, 7>
        (l2_out1, l2_out0, l20_c2_weight, l20_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l2_out0, "l20_c1_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    WRITE_TO_FILE_NAME(l2_out1, "l20_c2_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    // block 1
    tiled_conv
        <L2_DEPTH, L2_DEPTH, 3, 3, 1, 1,
        L2_SIDE, L2_SIDE, 64, 7, 7>
        (l2_out0, l2_out1, l21_c1_weight, l21_c1_bias, true);
    tiled_conv
        <L2_DEPTH, L2_DEPTH, 3, 3, 1, 1,
        L2_SIDE, L2_SIDE, 64, 7, 7>
        (l2_out1, l2_out0, l21_c2_weight, l21_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l2_out0, "l21_c1_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    WRITE_TO_FILE_NAME(l2_out1, "l21_c2_out", L2_DEPTH, L2_SIDE, L2_SIDE);

    // layer 3
    // downsample
    tiled_conv
        <L3_DEPTH, L2_DEPTH, 1, 1, 2, 0,
        L2_SIDE, L2_SIDE, 64, 14, 14>
        (l3_out1, l2_out1, l3_ds_weight, l3_ds_bias, false);
    WRITE_TO_FILE_NAME(l3_out1, "l3_ds_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    // block 0
    tiled_conv
        <L3_DEPTH, L2_DEPTH, 3, 3, 2, 1,
        L2_SIDE, L2_SIDE, 64, 14, 14>
        (l3_out0, l2_out1, l30_c1_weight, l30_c1_bias, true);
    tiled_conv
        <L3_DEPTH, L3_DEPTH, 3, 3, 1, 1,
        L3_SIDE, L3_SIDE, 64, 7, 7>
        (l3_out1, l3_out0, l30_c2_weight, l30_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l3_out0, "l30_c1_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    WRITE_TO_FILE_NAME(l3_out1, "l30_c2_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    // block 1
    tiled_conv
        <L3_DEPTH, L3_DEPTH, 3, 3, 1, 1,
        L3_SIDE, L3_SIDE, 64, 7, 7>
        (l3_out0, l3_out1, l31_c1_weight, l31_c1_bias, true);
    tiled_conv
        <L3_DEPTH, L3_DEPTH, 3, 3, 1, 1,
        L3_SIDE, L3_SIDE, 64, 7, 7>
        (l3_out1, l3_out0, l31_c2_weight, l31_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l3_out0, "l31_c1_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    WRITE_TO_FILE_NAME(l3_out1, "l31_c2_out", L3_DEPTH, L3_SIDE, L3_SIDE);


    // layer 4
    // downsample
    tiled_conv
        <L4_DEPTH, L3_DEPTH, 1, 1, 2, 0,
        L3_SIDE, L3_SIDE, 64, 14, 14>
        (l4_out1, l3_out1, l4_ds_weight, l4_ds_bias, false);
    WRITE_TO_FILE_NAME(l4_out1, "l4_ds_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    // block 0
    tiled_conv
        <L4_DEPTH, L3_DEPTH, 3, 3, 2, 1,
        L3_SIDE, L3_SIDE, 64, 14, 14>
        (l4_out0, l3_out1, l40_c1_weight, l40_c1_bias, true);
    tiled_conv
        <L4_DEPTH, L4_DEPTH, 3, 3, 1, 1,
        L4_SIDE, L4_SIDE, 64, 7, 7>
        (l4_out1, l4_out0, l40_c2_weight, l40_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l4_out0, "l40_c1_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    WRITE_TO_FILE_NAME(l4_out1, "l40_c2_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    // block 1
    tiled_conv
        <L4_DEPTH, L4_DEPTH, 3, 3, 1, 1,
        L4_SIDE, L4_SIDE, 64, 7, 7>
        (l4_out0, l4_out1, l41_c1_weight, l41_c1_bias, true);
    tiled_conv
        <L4_DEPTH, L4_DEPTH, 3, 3, 1, 1,
        L4_SIDE, L4_SIDE, 64, 7, 7>
        (l4_out1, l4_out0, l41_c2_weight, l41_c2_bias, true, true);
    WRITE_TO_FILE_NAME(l4_out0, "l41_c1_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    WRITE_TO_FILE_NAME(l4_out1, "l41_c2_out", L4_DEPTH, L4_SIDE, L4_SIDE);


    // avgpool
    avg_pool<L4_DEPTH, 7, 7>((fm_t (*)[7][7])l4_out1, avgpool_out);
    WRITE_TO_FILE(avgpool_out, AVG_POOL_SIZE, 1, 1);

    
    // fc
    linear_fc<L4_DEPTH, OUTPUT_SIZE>(avgpool_out, output, fc_weight, fc_bias);
    WRITE_TO_FILE(output, 1000, 1, 1);
}
