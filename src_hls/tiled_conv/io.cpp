#pragma once
#include "../util.h"

template<int BUF_DEPTH, int BUF_HEIGHT, int BUF_WIDTH,
         int FM_DEPTH, int FM_HEIGHT, int FM_WIDTH,
         int TILE_HEIGHT, int TILE_WIDTH, int PADDING>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const fm_t in_fm[FM_DEPTH][FM_HEIGHT][FM_WIDTH],
    const int  ti,
    const int  tj
)
{
    const int height_offset = ti * TILE_HEIGHT;
    const int width_offset  = tj * TILE_WIDTH;

    const int P = PADDING;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < BUF_DEPTH; c++) // FM and BUF have same depth
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
                //
                // Hint: Either load 0 or input feature into
                //       the buffer based on border conditions
                //in_fm_buf[f][i][j] = 0; // Just a placeholder

                int idx_h = height_offset + i - P;
                int idx_w = width_offset + j - P;

                if ((idx_h < 0 || idx_h >= FM_HEIGHT) ||
                    (idx_w < 0 || idx_w >= FM_WIDTH))
                {
                    in_fm_buf[c][i][j] = (fm_t) 0;
                }
                else
                {
                    //int dram_idx = idx_w + idx_h*FM_WIDTH + c*FM_WIDTH*FM_HEIGHT;
                    //in_fm_buf[c][i][j] = dram[dram_idx];
                    in_fm_buf[c][i][j] = in_fm[c][idx_h][idx_w];
                }

            }
        }
    }
}


template<int OUT_BUF_DEPTH, int IN_BUF_DEPTH, int KERNEL_HEIGHT, int KERNEL_WIDTH,
         int OUT_FM_DEPTH, int IN_FM_DEPTH>
void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    const wt_t weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t bias[OUT_FM_DEPTH],
    const int kernel_group
)
{
    const int kernel_offset  = kernel_group * OUT_BUF_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 7; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 7; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][c][kh][kw];
                }
            }
        }
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }

}

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH,
         int OUT_FM_DEPTH, int OUT_FM_HEIGHT, int OUT_FM_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH],
    const fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const int  ti,
    const int  tj,
    const int  kernel_group,
    const bool relu
)
{
    const int depth_offset  = kernel_group * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                //int dram_idx = (width_offset + j) + (height_offset + i)*OUT_FM_WIDTH + (depth_offset + f)*OUT_FM_WIDTH*OUT_FM_HEIGHT;
                //std::cout << f << ", " << i << ", " << j << std::endl;

                // ReLU in-place
                if(relu & (out_fm_buf[f][i][j] < (fm_t) 0))
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t) 0;
                    //out_fm[dram_idx] = (fm_t) 0;
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                    //out_fm[dram_idx] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}
