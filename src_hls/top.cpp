#include "conv.h"
#include "resnet18.cpp"

void load_from_DRAM(wt_t *DRAM, wt_t *local, int size) {
    for (int i = 0; i < size; i++) {
    #pragma HLS unroll 1
        local[i] = DRAM[i];
    }
}

#define LOAD_FROM_DRAM(x) load_from_DRAM((wt_t*) DRAM_##x, (wt_t*) x, sizeof(x) / sizeof(wt_t));

void top(
        fm_t DRAM_input[3][224][224],
        fm_t DRAM_output[1000],
        wt_t DRAM_conv1_weight[64][3][7][7],
        wt_t DRAM_conv1_bias[64],
        // layer 1
        wt_t DRAM_l10_c1_weight[64][64][3][3],
        wt_t DRAM_l10_c1_bias[64],
        wt_t DRAM_l10_c2_weight[64][64][3][3],
        wt_t DRAM_l10_c2_bias[64],
        wt_t DRAM_l11_c1_weight[64][64][3][3],
        wt_t DRAM_l11_c1_bias[64],
        wt_t DRAM_l11_c2_weight[64][64][3][3],
        wt_t DRAM_l11_c2_bias[64],
        // layer 2
        wt_t DRAM_l2_ds_weight[128][64][1][1],
        wt_t DRAM_l2_ds_bias[128],
        wt_t DRAM_l20_c1_weight[128][64][3][3],
        wt_t DRAM_l20_c1_bias[128],
        wt_t DRAM_l20_c2_weight[128][128][3][3],
        wt_t DRAM_l20_c2_bias[128],
        wt_t DRAM_l21_c1_weight[128][128][3][3],
        wt_t DRAM_l21_c1_bias[128],
        wt_t DRAM_l21_c2_weight[128][128][3][3],
        wt_t DRAM_l21_c2_bias[128],
        // layer 3
        wt_t DRAM_l3_ds_weight[256][128][1][1],
        wt_t DRAM_l3_ds_bias[256],
        wt_t DRAM_l30_c1_weight[256][128][3][3],
        wt_t DRAM_l30_c1_bias[256],
        wt_t DRAM_l30_c2_weight[256][256][3][3],
        wt_t DRAM_l30_c2_bias[256],
        wt_t DRAM_l31_c1_weight[256][256][3][3],
        wt_t DRAM_l31_c1_bias[256],
        wt_t DRAM_l31_c2_weight[256][256][3][3],
        wt_t DRAM_l31_c2_bias[256],
        // layer 4
        wt_t DRAM_l4_ds_weight[512][256][1][1],
        wt_t DRAM_l4_ds_bias[512],
        wt_t DRAM_l40_c1_weight[512][256][3][3],
        wt_t DRAM_l40_c1_bias[512],
        wt_t DRAM_l40_c2_weight[512][512][3][3],
        wt_t DRAM_l40_c2_bias[512],
        wt_t DRAM_l41_c1_weight[512][512][3][3],
        wt_t DRAM_l41_c1_bias[512],
        wt_t DRAM_l41_c2_weight[512][512][3][3],
        wt_t DRAM_l41_c2_bias[512],
        // fc
        wt_t DRAM_fc_weight[1000][512],
        wt_t DRAM_fc_bias[1000]
        )
{

        fm_t input[3][224][224];
        fm_t output[1000];
        wt_t conv1_weight[64][3][7][7];
        wt_t conv1_bias[64];
        // layer 1
        wt_t l10_c1_weight[64][64][3][3];
        wt_t l10_c1_bias[64];
        wt_t l10_c2_weight[64][64][3][3];
        wt_t l10_c2_bias[64];
        wt_t l11_c1_weight[64][64][3][3];
        wt_t l11_c1_bias[64];
        wt_t l11_c2_weight[64][64][3][3];
        wt_t l11_c2_bias[64];
        // layer 2
        wt_t l2_ds_weight[128][64][1][1];
        wt_t l2_ds_bias[128];
        wt_t l20_c1_weight[128][64][3][3];
        wt_t l20_c1_bias[128];
        wt_t l20_c2_weight[128][128][3][3];
        wt_t l20_c2_bias[128];
        wt_t l21_c1_weight[128][128][3][3];
        wt_t l21_c1_bias[128];
        wt_t l21_c2_weight[128][128][3][3];
        wt_t l21_c2_bias[128];
        // layer 3
        wt_t l3_ds_weight[256][128][1][1];
        wt_t l3_ds_bias[256];
        wt_t l30_c1_weight[256][128][3][3];
        wt_t l30_c1_bias[256];
        wt_t l30_c2_weight[256][256][3][3];
        wt_t l30_c2_bias[256];
        wt_t l31_c1_weight[256][256][3][3];
        wt_t l31_c1_bias[256];
        wt_t l31_c2_weight[256][256][3][3];
        wt_t l31_c2_bias[256];
        // layer 4
        wt_t l4_ds_weight[512][256][1][1];
        wt_t l4_ds_bias[512];
        wt_t l40_c1_weight[512][256][3][3];
        wt_t l40_c1_bias[512];
        wt_t l40_c2_weight[512][512][3][3];
        wt_t l40_c2_bias[512];
        wt_t l41_c1_weight[512][512][3][3];
        wt_t l41_c1_bias[512];
        wt_t l41_c2_weight[512][512][3][3];
        wt_t l41_c2_bias[512];
        // fc
        wt_t fc_weight[1000][512];
        wt_t fc_bias[1000];

    LOAD_FROM_DRAM(input);
    LOAD_FROM_DRAM(conv1_weight);
    LOAD_FROM_DRAM(conv1_bias);
    LOAD_FROM_DRAM(l10_c1_weight);
    LOAD_FROM_DRAM(l10_c1_bias);
    LOAD_FROM_DRAM(l10_c2_weight);
    LOAD_FROM_DRAM(l10_c2_bias);
    LOAD_FROM_DRAM(l11_c1_weight);
    LOAD_FROM_DRAM(l11_c1_bias);
    LOAD_FROM_DRAM(l11_c2_weight);
    LOAD_FROM_DRAM(l11_c2_bias);
    LOAD_FROM_DRAM(l2_ds_weight);
    LOAD_FROM_DRAM(l2_ds_bias);
    LOAD_FROM_DRAM(l20_c1_weight);
    LOAD_FROM_DRAM(l20_c1_bias);
    LOAD_FROM_DRAM(l20_c2_weight);
    LOAD_FROM_DRAM(l20_c2_bias);
    LOAD_FROM_DRAM(l21_c1_weight);
    LOAD_FROM_DRAM(l21_c1_bias);
    LOAD_FROM_DRAM(l21_c2_weight);
    LOAD_FROM_DRAM(l21_c2_bias);
    LOAD_FROM_DRAM(l3_ds_weight);
    LOAD_FROM_DRAM(l3_ds_bias);
    LOAD_FROM_DRAM(l30_c1_weight);
    LOAD_FROM_DRAM(l30_c1_bias);
    LOAD_FROM_DRAM(l30_c2_weight);
    LOAD_FROM_DRAM(l30_c2_bias);
    LOAD_FROM_DRAM(l31_c1_weight);
    LOAD_FROM_DRAM(l31_c1_bias);
    LOAD_FROM_DRAM(l31_c2_weight);
    LOAD_FROM_DRAM(l31_c2_bias);
    LOAD_FROM_DRAM(l4_ds_weight);
    LOAD_FROM_DRAM(l4_ds_bias);
    LOAD_FROM_DRAM(l40_c1_weight);
    LOAD_FROM_DRAM(l40_c1_bias);
    LOAD_FROM_DRAM(l40_c2_weight);
    LOAD_FROM_DRAM(l40_c2_bias);
    LOAD_FROM_DRAM(l41_c1_weight);
    LOAD_FROM_DRAM(l41_c1_bias);
    LOAD_FROM_DRAM(l41_c2_weight);
    LOAD_FROM_DRAM(l41_c2_bias);
    LOAD_FROM_DRAM(fc_weight);
    LOAD_FROM_DRAM(fc_bias);

    resnet18(
        input,
        DRAM_output,
        conv1_weight,
        conv1_bias,
        l10_c1_weight,
        l10_c1_bias,
        l10_c2_weight,
        l10_c2_bias,
        l11_c1_weight,
        l11_c1_bias,
        l11_c2_weight,
        l11_c2_bias,
        l2_ds_weight,
        l2_ds_bias,
        l20_c1_weight,
        l20_c1_bias,
        l20_c2_weight,
        l20_c2_bias,
        l21_c1_weight,
        l21_c1_bias,
        l21_c2_weight,
        l21_c2_bias,
        l3_ds_weight,
        l3_ds_bias,
        l30_c1_weight,
        l30_c1_bias,
        l30_c2_weight,
        l30_c2_bias,
        l31_c1_weight,
        l31_c1_bias,
        l31_c2_weight,
        l31_c2_bias,
        l4_ds_weight,
        l4_ds_bias,
        l40_c1_weight,
        l40_c1_bias,
        l40_c2_weight,
        l40_c2_bias,
        l41_c1_weight,
        l41_c1_bias,
        l41_c2_weight,
        l41_c2_bias,
        fc_weight,
        fc_bias
    );

}
