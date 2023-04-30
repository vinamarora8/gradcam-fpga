#include "../util.h"
#include "conv.cpp"
#include "io.cpp"
#include <iostream>
#include <cassert>

template<
int OUT_FM_DEPTH, int IN_FM_DEPTH, int KERNEL_HEIGHT, int KERNEL_WIDTH, int STRIDE, int PADDING, // Kernel
int IN_FM_HEIGHT, int IN_FM_WIDTH,  // Input
int OUT_BUF_DEPTH, int TILE_HEIGHT, int TILE_WIDTH // Tile shapes
>
void tiled_conv (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t layer_bias[OUT_FM_DEPTH],
    const bool relu,
    const bool inplace_residual = false
)
{
    static_assert(TILE_HEIGHT % STRIDE == 0, "TILE_HEIGHT must be a multiple of STRIDE");
    static_assert(TILE_WIDTH % STRIDE == 0, "TILE_WIDTH must be a multiple of STRIDE");
    static_assert(IN_FM_HEIGHT % STRIDE == 0, "IN_FM_HEIGHT must be a multiple of STRIDE");
    static_assert(IN_FM_WIDTH % STRIDE == 0, "IN_FM_WIDTH must be a multiple of STRIDE");
    static_assert(OUT_FM_DEPTH % OUT_BUF_DEPTH == 0, "OUT_FM_DEPTH must be a multiple of OUT_BUF_DEPTH");
    static_assert(IN_FM_HEIGHT % TILE_HEIGHT == 0, "IN_FM_HEIGHT must be a multiple of TILE_HEIGHT");
    static_assert(IN_FM_WIDTH % TILE_WIDTH == 0, "IN_FM_WIDTH must be a multiple of TILE_WIDTH");

    const int MARGIN = 2 * PADDING;
    const int IN_BUF_DEPTH = IN_FM_DEPTH;
    const int IN_BUF_HEIGHT = TILE_HEIGHT + MARGIN;
    const int IN_BUF_WIDTH = TILE_WIDTH + MARGIN;
    const int OUT_BUF_HEIGHT = TILE_HEIGHT / STRIDE;
    const int OUT_BUF_WIDTH = TILE_WIDTH / STRIDE;
    const int OUT_FM_HEIGHT = IN_FM_HEIGHT / STRIDE;
    const int OUT_FM_WIDTH = IN_FM_WIDTH / STRIDE;

    const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;

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
    static fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
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

            load_fm_tile_block_from_DRAM
                <IN_BUF_DEPTH, TILE_HEIGHT, TILE_WIDTH, PADDING,
                IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH>
                (conv_in_buf, input_feature_map, ti, tj, 0);

            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GRPS; tk++)
            {

                if (inplace_residual)
                {
                    load_fm_tile_block_from_DRAM
                        <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, 0,
                        OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH>
                        (conv_out_buf, output_feature_map, ti, tj, tk);
                }
                else
                {
                    for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        for (int i = 0; i < OUT_BUF_HEIGHT; i++)
                            for (int j = 0; j < OUT_BUF_WIDTH; j++)
                                conv_out_buf[f][i][j] = (fm_t) 0;
                }

                load_layer_params_from_DRAM
                    <OUT_BUF_DEPTH, IN_BUF_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH,
                    OUT_FM_DEPTH, IN_FM_DEPTH>
                    (conv_wt_buf, conv_bias_buf, layer_weights, layer_bias, tk);

                conv_small
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH,
                    IN_BUF_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH,
                    KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE>
                    (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf);

                store_output_tile_to_DRAM
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH,
                    OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH>
                    (output_feature_map, conv_out_buf, ti, tj, tk, relu);

            }
        }
    }
}

