#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#define CSIM_DEBUG

#include "util.h"
#include "conv.h"

// Floating point precision arrays
float input[3][224][224],
float output[1000],
float conv1_weight[64][3][7][7],
float conv1_bias[64],
// layer 1
float l10_c1_weight[64][64][3][3],
float l10_c1_bias[64],
float l10_c2_weight[64][64][3][3],
float l10_c2_bias[64],
float l11_c1_weight[64][64][3][3],
float l11_c1_bias[64],
float l11_c2_weight[64][64][3][3],
float l11_c2_bias[64],
// layer 2
float l2_ds_weight[128][64][1][1],
float l2_ds_bias[128],
float l20_c1_weight[128][64][3][3],
float l20_c1_bias[128],
float l20_c2_weight[128][128][3][3],
float l20_c2_bias[128],
float l21_c1_weight[128][128][3][3],
float l21_c1_bias[128],
float l21_c2_weight[128][128][3][3],
float l21_c2_bias[128],
// layer 3
float l3_ds_weight[256][128][1][1],
float l3_ds_bias[256],
float l30_c1_weight[256][128][3][3],
float l30_c1_bias[256],
float l30_c2_weight[256][256][3][3],
float l30_c2_bias[256],
float l31_c1_weight[256][256][3][3],
float l31_c1_bias[256],
float l31_c2_weight[256][256][3][3],
float l31_c2_bias[256],
// layer 4
float l4_ds_weight[512][256][1][1],
float l4_ds_bias[512],
float l40_c1_weight[512][256][3][3],
float l40_c1_bias[512],
float l40_c2_weight[512][512][3][3],
float l40_c2_bias[512],
float l41_c1_weight[512][512][3][3],
float l41_c1_bias[512],
float l41_c2_weight[512][512][3][3],
float l41_c2_bias[512],
// fc
float fc_weight[1000][512],
float fc_bias[1000]

// Fixed point precision
fm_t fixp_input[3][224][224],
fm_t fixp_output[1000],
wt_t fixp_conv1_weight[64][3][7][7],
wt_t fixp_conv1_bias[64],
// layer 1
wt_t fixp_l10_c1_weight[64][64][3][3],
wt_t fixp_l10_c1_bias[64],
wt_t fixp_l10_c2_weight[64][64][3][3],
wt_t fixp_l10_c2_bias[64],
wt_t fixp_l11_c1_weight[64][64][3][3],
wt_t fixp_l11_c1_bias[64],
wt_t fixp_l11_c2_weight[64][64][3][3],
wt_t fixp_l11_c2_bias[64],
// layer 2
wt_t fixp_l2_ds_weight[128][64][1][1],
wt_t fixp_l2_ds_bias[128],
wt_t fixp_l20_c1_weight[128][64][3][3],
wt_t fixp_l20_c1_bias[128],
wt_t fixp_l20_c2_weight[128][128][3][3],
wt_t fixp_l20_c2_bias[128],
wt_t fixp_l21_c1_weight[128][128][3][3],
wt_t fixp_l21_c1_bias[128],
wt_t fixp_l21_c2_weight[128][128][3][3],
wt_t fixp_l21_c2_bias[128],
// layer 3
wt_t fixp_l3_ds_weight[256][128][1][1],
wt_t fixp_l3_ds_bias[256],
wt_t fixp_l30_c1_weight[256][128][3][3],
wt_t fixp_l30_c1_bias[256],
wt_t fixp_l30_c2_weight[256][256][3][3],
wt_t fixp_l30_c2_bias[256],
wt_t fixp_l31_c1_weight[256][256][3][3],
wt_t fixp_l31_c1_bias[256],
wt_t fixp_l31_c2_weight[256][256][3][3],
wt_t fixp_l31_c2_bias[256],
// layer 4
wt_t fixp_l4_ds_weight[512][256][1][1],
wt_t fixp_l4_ds_bias[512],
wt_t fixp_l40_c1_weight[512][256][3][3],
wt_t fixp_l40_c1_bias[512],
wt_t fixp_l40_c2_weight[512][512][3][3],
wt_t fixp_l40_c2_bias[512],
wt_t fixp_l41_c1_weight[512][512][3][3],
wt_t fixp_l41_c1_bias[512],
wt_t fixp_l41_c2_weight[512][512][3][3],
wt_t fixp_l41_c2_bias[512],
// fc
wt_t fixp_fc_weight[1000][512],
wt_t fixp_fc_bias[1000]

void load_from_files(){
    std::string root_dir = "bin/";

    load_fp_and_fixp_vals(input, fixp_input, root_dir + VAR_NAME(input), {3,224,224});
    load_fp_and_fixp_vals(output, fixp_output, root_dir + VAR_NAME(output), {1000});
    load_fp_and_fixp_vals(conv1_weight, fixp_conv1_weight, root_dir + VAR_NAME(conv1_weight), {64,3,7,7});
    load_fp_and_fixp_vals(conv1_bias, fixp_conv1_bias, root_dir + VAR_NAME(conv1_bias), {64});
    load_fp_and_fixp_vals(l10_c1_weight, fixp_l10_c1_weight, root_dir + VAR_NAME(l10_c1_weight), {64,64,3,3});
    load_fp_and_fixp_vals(l10_c1_bias, fixp_l10_c1_bias, root_dir + VAR_NAME(l10_c1_bias), {64});
    load_fp_and_fixp_vals(l10_c2_weight, fixp_l10_c2_weight, root_dir + VAR_NAME(l10_c2_weight), {64,64,3,3});
    load_fp_and_fixp_vals(l10_c2_bias, fixp_l10_c2_bias, root_dir + VAR_NAME(l10_c2_bias), {64});
    load_fp_and_fixp_vals(l11_c1_weight, fixp_l11_c1_weight, root_dir + VAR_NAME(l11_c1_weight), {64,64,3,3});
    load_fp_and_fixp_vals(l11_c1_bias, fixp_l11_c1_bias, root_dir + VAR_NAME(l11_c1_bias), {64});
    load_fp_and_fixp_vals(l11_c2_weight, fixp_l11_c2_weight, root_dir + VAR_NAME(l11_c2_weight), {64,64,3,3});
    load_fp_and_fixp_vals(l11_c2_bias, fixp_l11_c2_bias, root_dir + VAR_NAME(l11_c2_bias), {64});
    load_fp_and_fixp_vals(l2_ds_weight, fixp_l2_ds_weight, root_dir + VAR_NAME(l2_ds_weight), {128,64,1,1});
    load_fp_and_fixp_vals(l2_ds_bias, fixp_l2_ds_bias, root_dir + VAR_NAME(l2_ds_bias), {128});
    load_fp_and_fixp_vals(l20_c1_weight, fixp_l20_c1_weight, root_dir + VAR_NAME(l20_c1_weight), {128,64,3,3});
    load_fp_and_fixp_vals(l20_c1_bias, fixp_l20_c1_bias, root_dir + VAR_NAME(l20_c1_bias), {128});
    load_fp_and_fixp_vals(l20_c2_weight, fixp_l20_c2_weight, root_dir + VAR_NAME(l20_c2_weight), {128,128,3,3});
    load_fp_and_fixp_vals(l20_c2_bias, fixp_l20_c2_bias, root_dir + VAR_NAME(l20_c2_bias), {128});
    load_fp_and_fixp_vals(l21_c1_weight, fixp_l21_c1_weight, root_dir + VAR_NAME(l21_c1_weight), {128,128,3,3});
    load_fp_and_fixp_vals(l21_c1_bias, fixp_l21_c1_bias, root_dir + VAR_NAME(l21_c1_bias), {128});
    load_fp_and_fixp_vals(l21_c2_weight, fixp_l21_c2_weight, root_dir + VAR_NAME(l21_c2_weight), {128,128,3,3});
    load_fp_and_fixp_vals(l21_c2_bias, fixp_l21_c2_bias, root_dir + VAR_NAME(l21_c2_bias), {128});
    load_fp_and_fixp_vals(l3_ds_weight, fixp_l3_ds_weight, root_dir + VAR_NAME(l3_ds_weight), {256,128,1,1});
    load_fp_and_fixp_vals(l3_ds_bias, fixp_l3_ds_bias, root_dir + VAR_NAME(l3_ds_bias), {256});
    load_fp_and_fixp_vals(l30_c1_weight, fixp_l30_c1_weight, root_dir + VAR_NAME(l30_c1_weight), {256,128,3,3});
    load_fp_and_fixp_vals(l30_c1_bias, fixp_l30_c1_bias, root_dir + VAR_NAME(l30_c1_bias), {256});
    load_fp_and_fixp_vals(l30_c2_weight, fixp_l30_c2_weight, root_dir + VAR_NAME(l30_c2_weight), {256,256,3,3});
    load_fp_and_fixp_vals(l30_c2_bias, fixp_l30_c2_bias, root_dir + VAR_NAME(l30_c2_bias), {256});
    load_fp_and_fixp_vals(l31_c1_weight, fixp_l31_c1_weight, root_dir + VAR_NAME(l31_c1_weight), {256,256,3,3});
    load_fp_and_fixp_vals(l31_c1_bias, fixp_l31_c1_bias, root_dir + VAR_NAME(l31_c1_bias), {256});
    load_fp_and_fixp_vals(l31_c2_weight, fixp_l31_c2_weight, root_dir + VAR_NAME(l31_c2_weight), {256,256,3,3});
    load_fp_and_fixp_vals(l31_c2_bias, fixp_l31_c2_bias, root_dir + VAR_NAME(l31_c2_bias), {256});
    load_fp_and_fixp_vals(l4_ds_weight, fixp_l4_ds_weight, root_dir + VAR_NAME(l4_ds_weight), {512,256,1,1});
    load_fp_and_fixp_vals(l4_ds_bias, fixp_l4_ds_bias, root_dir + VAR_NAME(l4_ds_bias), {512});
    load_fp_and_fixp_vals(l40_c1_weight, fixp_l40_c1_weight, root_dir + VAR_NAME(l40_c1_weight), {512,256,3,3});
    load_fp_and_fixp_vals(l40_c1_bias, fixp_l40_c1_bias, root_dir + VAR_NAME(l40_c1_bias), {512});
    load_fp_and_fixp_vals(l40_c2_weight, fixp_l40_c2_weight, root_dir + VAR_NAME(l40_c2_weight), {512,512,3,3});
    load_fp_and_fixp_vals(l40_c2_bias, fixp_l40_c2_bias, root_dir + VAR_NAME(l40_c2_bias), {512});
    load_fp_and_fixp_vals(l41_c1_weight, fixp_l41_c1_weight, root_dir + VAR_NAME(l41_c1_weight), {512,512,3,3});
    load_fp_and_fixp_vals(l41_c1_bias, fixp_l41_c1_bias, root_dir + VAR_NAME(l41_c1_bias), {512});
    load_fp_and_fixp_vals(l41_c2_weight, fixp_l41_c2_weight, root_dir + VAR_NAME(l41_c2_weight), {512,512,3,3});
    load_fp_and_fixp_vals(l41_c2_bias, fixp_l41_c2_bias, root_dir + VAR_NAME(l41_c2_bias), {512});
    load_fp_and_fixp_vals(fc_weight, fixp_fc_weight, root_dir + VAR_NAME(fc_weight), {1000,512});
    load_fp_and_fixp_vals(fc_bias, fixp_fc_bias, root_dir + VAR_NAME(fc_bias), {1000});
}

int main(int argc, char *argv[])
{

    load_from_files();
 
    resnet18(
        fixp_input,
        fixp_output,
        fixp_conv1_weight,
        fixp_conv1_bias,
        fixp_l10_c1_weight,
        fixp_l10_c1_bias,
        fixp_l10_c2_weight,
        fixp_l10_c2_bias,
        fixp_l11_c1_weight,
        fixp_l11_c1_bias,
        fixp_l11_c2_weight,
        fixp_l11_c2_bias,
        fixp_l2_ds_weight,
        fixp_l2_ds_bias,
        fixp_l20_c1_weight,
        fixp_l20_c1_bias,
        fixp_l20_c2_weight,
        fixp_l20_c2_bias,
        fixp_l21_c1_weight,
        fixp_l21_c1_bias,
        fixp_l21_c2_weight,
        fixp_l21_c2_bias,
        fixp_l3_ds_weight,
        fixp_l3_ds_bias,
        fixp_l30_c1_weight,
        fixp_l30_c1_bias,
        fixp_l30_c2_weight,
        fixp_l30_c2_bias,
        fixp_l31_c1_weight,
        fixp_l31_c1_bias,
        fixp_l31_c2_weight,
        fixp_l31_c2_bias,
        fixp_l4_ds_weight,
        fixp_l4_ds_bias,
        fixp_l40_c1_weight,
        fixp_l40_c1_bias,
        fixp_l40_c2_weight,
        fixp_l40_c2_bias,
        fixp_l41_c1_weight,
        fixp_l41_c1_bias,
        fixp_l41_c2_weight,
        fixp_l41_c2_bias,
        fixp_fc_weight,
        fixp_fc_bias
    )

    return 0;
}
