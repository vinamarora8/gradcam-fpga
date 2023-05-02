#include "../util.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace conv_ds
{

const int IN_BUF_DEPTH = 32;
const int OUT_BUF_DEPTH = 32;
const int BUF_HEIGHT = 7;
const int BUF_WIDTH = 7;

void tiled_conv_ds_core(
    fm_t out_feature_map[],
    const fm_t in_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int N_TILE_ROWS,
    const int N_TILE_COLS,
    const int N_TILE_LAYERS,
    const int KERNEL_GROUPS
);

template<int OUT_FM_DEPTH, int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH>
void tiled_conv_ds(
    fm_t out_feature_map[OUT_FM_DEPTH][IN_FM_HEIGHT/2][IN_FM_WIDTH/2],
    const fm_t in_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    const wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH],
    const wt_t layer_bias[OUT_FM_DEPTH]
)
{
    const int N_TILE_ROWS = IN_FM_HEIGHT / (2 * BUF_HEIGHT);
    const int N_TILE_COLS = IN_FM_WIDTH / (2 * BUF_WIDTH);
    const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;
    const int KERNEL_GROUPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;

    // Power of 2 checks on TILE_ROWS, TILE_COLS, TILE_LAYERS, KERNEL_GROUPS
    static_assert(N_TILE_ROWS > 0 && (N_TILE_ROWS & (N_TILE_ROWS - 1)) == 0);
    static_assert(N_TILE_COLS > 0 && (N_TILE_COLS & (N_TILE_COLS - 1)) == 0);
    static_assert(N_TILE_LAYERS > 0 && (N_TILE_LAYERS & (N_TILE_LAYERS - 1)) == 0);
    static_assert(KERNEL_GROUPS > 0 && (KERNEL_GROUPS & (KERNEL_GROUPS - 1)) == 0);

    // Find log2 of above:
    const int LOG_TILE_ROWS = log2(N_TILE_ROWS);
    const int LOG_TILE_COLS = log2(N_TILE_COLS);
    const int LOG_TILE_LAYERS = log2(N_TILE_LAYERS);
    const int LOG_KERNEL_GROUPS = log2(KERNEL_GROUPS);

    tiled_conv_ds_core(
        (fm_t *) out_feature_map,
        (fm_t *) in_feature_map,
        (fm_t *) layer_weights,
        layer_bias,
        LOG_TILE_ROWS,
        LOG_TILE_COLS,
        LOG_TILE_LAYERS,
        LOG_KERNEL_GROUPS
    );
}

void conv(
    fm_t out_buf[OUT_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const fm_t in_buf[IN_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const wt_t wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH],
    const wt_t bias_buf[OUT_BUF_DEPTH],
    const int tl
)
{
    #pragma HLS inline off

CONV_IN_D: for (int c = 0; c < IN_BUF_DEPTH; c++)
    CONV_ROW: for (int i = 0; i < BUF_HEIGHT; i++)
        CONV_COL: for (int j = 0; j < BUF_WIDTH; j++)
            CONV_OUT_D: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                {
                    #pragma HLS pipeline II=1
                    fm_t x = out_buf[f][i][j];
                    if (c == 0 && tl == 0)
                        x = bias_buf[f] + in_buf[c][i][j] * wt_buf[f][c];
                    else
                        x += in_buf[c][i][j] * wt_buf[f][c];

                    out_buf[f][i][j] = x;
                }
}

void tiled_conv_ds_core(
    fm_t out_feature_map[],
    const fm_t in_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int LOG_TILE_ROWS,
    const int LOG_TILE_COLS,
    const int LOG_TILE_LAYERS,
    const int LOG_KERNEL_GROUPS
)
{
    #pragma HLS INTERFACE m_axi depth=1  port=in_feature_map   bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights    bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias       bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=out_feature_map  bundle=d_
    #pragma HLS INTERFACE s_axilite register	port=return

    const int N_TILE_ROWS = 1 << LOG_TILE_ROWS;
    const int N_TILE_COLS = 1 << LOG_TILE_COLS;
    const int N_TILE_LAYERS = 1 << LOG_TILE_LAYERS;
    const int KERNEL_GROUPS = 1 << LOG_KERNEL_GROUPS;

    const int IN_FM_DEPTH = IN_BUF_DEPTH * N_TILE_LAYERS;
    const int IN_FM_WIDTH = 2 * BUF_WIDTH * N_TILE_COLS;
    const int IN_FM_HEIGHT = 2 * BUF_HEIGHT * N_TILE_ROWS;
    const int OUT_FM_DEPTH = OUT_BUF_DEPTH * KERNEL_GROUPS;
    const int OUT_FM_HEIGHT = IN_FM_HEIGHT / 2;
    const int OUT_FM_WIDTH = IN_FM_WIDTH / 2;

    static fm_t in_buf[IN_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH];
    static fm_t out_buf[OUT_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH];
    static fm_t wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH];
    static fm_t bias_buf[OUT_BUF_DEPTH];

    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GROUPS; tk++)
            {
                    // Load layer bias
                BIAS: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        bias_buf[f] = layer_bias[tk*OUT_BUF_DEPTH + f];

                TILE_LYR:
                for (int tl = 0; tl < N_TILE_LAYERS; tl++)
                {
                    // Load input tile
                INP_D: for (int c = 0; c < IN_BUF_DEPTH; c++)
                        INP_R: for (int i = 0; i < BUF_HEIGHT; i++)
                            INP_C: for (int j = 0; j < BUF_WIDTH; j++)
                            {
                                #pragma HLS pipeline II=1
                                int idx_d = tl*IN_BUF_DEPTH + c;
                                int idx_h = 2*ti*BUF_HEIGHT + 2*i;
                                int idx_w = 2*tj*BUF_WIDTH + 2*j;
                                in_buf[c][i][j] = in_feature_map[idx_d*IN_FM_HEIGHT*IN_FM_WIDTH + idx_h*IN_FM_WIDTH + idx_w];
                            }

                    // Load layer weights
                KER_OUT: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        KER_IN: for (int c = 0; c < IN_BUF_DEPTH; c++)
                        {
                            #pragma HLS pipeline II=1
                            wt_buf[f][c] = layer_weights[(tk*OUT_BUF_DEPTH + f)*IN_FM_DEPTH + tl*IN_BUF_DEPTH + c];
                        }

                    // Compute
                     conv_ds::conv(out_buf, in_buf, wt_buf, bias_buf, tl);
                }

                // Store output tile
            OUT_D: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                    OUT_R: for (int i = 0; i < BUF_HEIGHT; i++)
                        OUT_C: for (int j = 0; j < BUF_WIDTH; j++)
                        {
                            #pragma HLS pipeline II=1
                            int idx_d = tk*OUT_BUF_DEPTH + f;
                            int idx_h = ti*BUF_HEIGHT + i;
                            int idx_w = tj*BUF_WIDTH + j;
                            out_feature_map[idx_d*OUT_FM_HEIGHT*OUT_FM_WIDTH + idx_h*OUT_FM_WIDTH + idx_w] = out_buf[f][i][j];
                        }
            }
            
        }
    }
}


void test_conv(
    fm_t out_feature_map[128][28][28],
    fm_t in_feature_map[64][56][56],
    wt_t layer_weights[128][64],
    wt_t layer_bias[128]
)
{
    conv_ds::tiled_conv_ds<128, 64, 56, 56>(out_feature_map, in_feature_map, layer_weights, layer_bias);
}

#if 0
#include "conv.cpp"
#include "io.cpp"

template<
int OUT_BUF_DEPTH, int IN_BUF_DEPTH, int KERNEL_HEIGHT, int KERNEL_WIDTH, int STRIDE, int PADDING, // Kernel
int TILE_HEIGHT, int TILE_WIDTH // Tile shapes
>
void tiled_conv_core (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int KERNEL_GRPS,
    const int N_TILE_LAYERS,
    const int N_TILE_ROWS,
    const int N_TILE_COLS,
    const bool relu,
    const bool inplace_residual = false
)
{
    static_assert(STRIDE == 1 || STRIDE == 2, "STRIDE > 2 not implemented");
    static_assert(TILE_HEIGHT % STRIDE == 0, "TILE_HEIGHT must be a multiple of STRIDE");
    static_assert(TILE_WIDTH % STRIDE == 0, "TILE_WIDTH must be a multiple of STRIDE");

    const int IN_FM_DEPTH = IN_BUF_DEPTH * N_TILE_LAYERS;
    const int IN_FM_HEIGHT = TILE_HEIGHT * N_TILE_ROWS;
    const int IN_FM_WIDTH = TILE_WIDTH * N_TILE_COLS;
    const int OUT_FM_DEPTH = OUT_BUF_DEPTH * KERNEL_GRPS;
    const int OUT_FM_HEIGHT = STRIDE == 1 ? IN_FM_HEIGHT : IN_FM_HEIGHT >> 1;
    const int OUT_FM_WIDTH = STRIDE == 1 ? IN_FM_WIDTH : IN_FM_WIDTH >> 1;


    assert(IN_FM_HEIGHT % STRIDE == 0);
    assert(IN_FM_WIDTH % STRIDE == 0);
    assert(OUT_FM_DEPTH % OUT_BUF_DEPTH == 0);
    assert(IN_FM_HEIGHT % TILE_HEIGHT == 0);
    assert(IN_FM_WIDTH % TILE_WIDTH == 0);


    const int MARGIN = 2 * PADDING;
    const int IN_BUF_HEIGHT = TILE_HEIGHT + MARGIN;
    const int IN_BUF_WIDTH = TILE_WIDTH + MARGIN;
    const int OUT_BUF_HEIGHT = STRIDE == 1 ? TILE_HEIGHT : TILE_HEIGHT >> 1;
    const int OUT_BUF_WIDTH = STRIDE == 1 ? TILE_WIDTH : TILE_WIDTH >> 1;

    /*
    const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;
    const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;
    */

    const fm_dims_s in_fm_dims = {IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH};
    const fm_dims_s out_fm_dims = {OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH};


    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS.
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights       bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias          bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm
    #pragma HLS INTERFACE s_axilite register	port=return

    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    static fm_t conv_in_buf[IN_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    static wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    static wt_t conv_bias_buf[OUT_BUF_DEPTH];
    static fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GRPS; tk++)
            {
                TILE_LYR:
                for (int tl = 0; tl < N_TILE_LAYERS; tl++)
                {
                    conv_ds::load_fm_tile_block_from_DRAM
                        <IN_BUF_DEPTH, TILE_HEIGHT, TILE_WIDTH>
                        (conv_in_buf, input_feature_map, in_fm_dims, ti, tj, tl);

                    conv_ds::load_layer_params_from_DRAM
                        <OUT_BUF_DEPTH, IN_BUF_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH>
                        (conv_wt_buf, conv_bias_buf, (fm_t *) layer_weights, layer_bias, 
                         OUT_FM_DEPTH, IN_FM_DEPTH, tk, tl);

                    bool residual = inplace_residual || (tl != 0);
                    conv_ds::conv_small
                        <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH,
                        IN_BUF_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH,
                        KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE>
                        (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, residual);

                }

                conv_ds::store_output_tile_to_DRAM
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH>
                    (output_feature_map, conv_out_buf, out_fm_dims, ti, tj, tk, relu);

            }
        }
    }
}

const int KERNEL_HEIGHT = 1;
const int KERNEL_WIDTH = 1;
const int STRIDE = 2;
const int PADDING = 0;
const int OUT_BUF_DEPTH = 64;
const int IN_BUF_DEPTH = 32;
const int TILE_HEIGHT = 14;
const int TILE_WIDTH = 14;

template<int OUT_FM_DEPTH, int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH>
inline void tiled_conv (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t layer_bias[OUT_FM_DEPTH],
    const bool relu,
    const bool inplace_residual = false
)
{ 

    static_assert(IN_FM_DEPTH >= IN_BUF_DEPTH, "IN_FM_WIDTH >= IN_BUF_DEPTH");

    const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;
    const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;

    conv_ds::tiled_conv_core
        <OUT_BUF_DEPTH, IN_BUF_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE, PADDING,
        TILE_HEIGHT, TILE_WIDTH>
        (output_feature_map,
        input_feature_map,
        (fm_t *) layer_weights,
        (fm_t *) layer_bias,
        KERNEL_GRPS,
        N_TILE_LAYERS,
        N_TILE_ROWS,
        N_TILE_COLS,
        relu,
        inplace_residual
        );

}
#endif
        

}
