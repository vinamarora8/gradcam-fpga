#include "../util.h"
#include <iostream>
#include <cassert>

namespace conv_3x3_s1 {

#include "params.hpp"
#include "conv.cpp"
#include "io.cpp"

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
    #pragma HLS inline off
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights       bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias          bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm_out
    #pragma HLS INTERFACE s_axilite register	port=return

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
            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GRPS; tk++)
            {
                if (inplace_residual)
                {
                    conv_3x3_s1::load_fm_tile_block_from_DRAM
                        <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, 0>
                        (conv_out_buf, output_feature_map, OUT_FM_HEIGHT, OUT_FM_WIDTH, ti, tj, tk);
                }

                TILE_LYR:
                for (int tl = 0; tl < N_TILE_LAYERS; tl++)
                {
                    conv_3x3_s1::load_fm_tile_block_from_DRAM
                        <IN_BUF_DEPTH, TILE_HEIGHT, TILE_WIDTH, PADDING>
                        (conv_in_buf, input_feature_map, IN_FM_HEIGHT, IN_FM_WIDTH, ti, tj, tl);

                    conv_3x3_s1::load_layer_params_from_DRAM
                        (conv_wt_buf, conv_bias_buf, (fm_t *) layer_weights, layer_bias, 
                         OUT_FM_DEPTH, IN_FM_DEPTH, tk, tl);

                    bool residual = inplace_residual || (tl != 0);
                    conv_3x3_s1::conv_small
                        (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, residual);

                }

                conv_3x3_s1::store_output_tile_to_DRAM
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH>
                    (output_feature_map, conv_out_buf, out_fm_dims, ti, tj, tk, relu);

            }
        }
    }
}


template<int OUT_FM_DEPTH, int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH>
inline void tiled_conv (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const bool relu,
    const bool inplace_residual = false
)
{ 
    #pragma HLS inline

    static_assert(IN_FM_DEPTH >= IN_BUF_DEPTH, "IN_FM_WIDTH >= IN_BUF_DEPTH");

    const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;
    const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;

    tiled_conv_core(
        output_feature_map,
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


void test_conv(
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[]
)
{
    conv_3x3_s1::tiled_conv<64, 64, 56, 56>(
        output_feature_map,
        input_feature_map,
        layer_weights,
        layer_bias,
        true,
        false
        );

    conv_3x3_s1::tiled_conv<64, 64, 56, 56>(
        output_feature_map,
        input_feature_map,
        layer_weights,
        layer_bias,
        true,
        true
        );
}
        
}
