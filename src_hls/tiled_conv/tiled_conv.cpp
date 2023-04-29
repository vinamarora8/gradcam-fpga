#include "util.h"
#include "conv.cpp"
#include "io.cpp"
#include <iostream>
#include <cassert>

template<
int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH,
int OUT_FM_DEPTH, int OUT_FM_HEIGHT, int OUT_FM_WIDTH,
int TILE_HEIGHT, int TILE_WIDTH,
int IN_BUF_DEPTH, int IN_BUF_HEIGHT, int IN_BUF_WIDTH,
int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH,
int KERNEL_HEIGHT, int KERNEL_WIDTH, int STRIDE, int PADDING>
void tiled_conv (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH],
    bool relu
)
{
    const int MARGIN = 2 * PADDING;
    /*
    const int IN_BUF_H = TILE_HEIGHT + MARGIN;
    const int IN_BUF_W = TILE_WIDTH + MARGIN;
    assert(TILE_HEIGHT % STRIDE == 0);
    assert(TILE_WIDTH % STRIDE == 0);
    const int OUT_BUF_H = TILE_HEIGHT / STRIDE;
    */

    std::cout << "OUT_BUF_HEIGHT: " << OUT_BUF_HEIGHT << std::endl;
    std::cout << "TILE_HEIGHT: " << TILE_HEIGHT << std::endl;
    std::cout << "STRIDE: " << STRIDE << std::endl;

    assert(IN_BUF_HEIGHT == (TILE_HEIGHT + MARGIN));
    assert(IN_BUF_WIDTH == (TILE_WIDTH + MARGIN));
    assert(OUT_BUF_HEIGHT == (TILE_HEIGHT / STRIDE));
    assert(OUT_BUF_WIDTH == (TILE_WIDTH / STRIDE));

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
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH] = {0};

    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;

    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;

            const int kernel_groups = OUT_FM_DEPTH / OUT_BUF_DEPTH;

            load_input_tile_block_from_DRAM
                <IN_BUF_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH,
                IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH,
                TILE_HEIGHT, TILE_WIDTH, PADDING>
                (conv_in_buf, input_feature_map, ti, tj);

            KERNEL_GRP:
            for (int tk = 0; tk < kernel_groups; tk++)
            {

                load_layer_params_from_DRAM
                    <OUT_BUF_DEPTH, IN_BUF_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH,
                    OUT_FM_DEPTH, IN_FM_DEPTH>
                    (conv_wt_buf, conv_bias_buf, layer_weights, layer_bias, tk);
                
                conv_small
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH,
                    IN_BUF_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH,
                    KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE, PADDING>
                    (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf);

                store_output_tile_to_DRAM
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH,
                    OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH>
                    (output_feature_map, conv_out_buf, ti, tj, tk, relu);

            }



        }
    }
}
