#pragma once
#include "../util.h"
#include "params.hpp"

template<int TILE_DEPTH, int TILE_HEIGHT, int TILE_WIDTH, int PADDING>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[TILE_DEPTH][TILE_HEIGHT + 2*PADDING][TILE_WIDTH + 2*PADDING],
    const fm_t in_fm[],
    const int FM_HEIGHT, const int FM_WIDTH,
    const int ti,
    const int tj,
    const int tk
)
{
    const int depth_offset = tk * TILE_DEPTH;
    const int height_offset = ti * TILE_HEIGHT;
    const int width_offset  = tj * TILE_WIDTH;

    const int BUF_DEPTH = TILE_DEPTH;
    const int BUF_HEIGHT = TILE_HEIGHT + 2*PADDING;
    const int BUF_WIDTH = TILE_WIDTH + 2*PADDING;

    const int P = PADDING;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < BUF_DEPTH; c++) // FM and BUF have same depth
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            for(int j = 0; j < BUF_WIDTH; j++)
            {
                #pragma HLS PIPELINE II=1
                int idx_w = width_offset - P + j;
                int idx_h = height_offset - P + i;
                int idx_d = depth_offset + c;
                int idx = (idx_w) + (idx_h)*FM_WIDTH + (idx_d)*FM_WIDTH*FM_HEIGHT;

                if ((idx_h < 0 || idx_h >= FM_HEIGHT) ||
                    (idx_w < 0 || idx_w >= FM_WIDTH))
                {
                    in_fm_buf[c][i][j] = (fm_t) 0;
                }
                else
                {
                    in_fm_buf[c][i][j] = in_fm[idx];
                }
            }
        }
    }
}


void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    const wt_t weights[],
    const wt_t bias[],
    const dim_t OUT_FM_DEPTH,
    const dim_t IN_FM_DEPTH,
    const int tk,
    const int tl
)
{
    const int kernel_offset  = tk * OUT_BUF_DEPTH;
    const int tile_layer_offset = tl * IN_BUF_DEPTH;

    int idx_f = kernel_offset*IN_FM_DEPTH*KERNEL_HEIGHT*KERNEL_WIDTH;
    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        int idx_c = tile_layer_offset*KERNEL_HEIGHT*KERNEL_WIDTH;
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            int idx_kh = 0;
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < KERNEL_HEIGHT; kh++)
	        {
                int idx_kw = 0;
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < KERNEL_WIDTH; kw++)
	            {
                    #pragma HLS PIPELINE II=1
                    int idx_f = (kernel_offset + f)*IN_FM_DEPTH*KERNEL_HEIGHT*KERNEL_WIDTH;
                    int idx_c = (tile_layer_offset + c)*KERNEL_HEIGHT*KERNEL_WIDTH;
                    int idx_kh = kh*KERNEL_WIDTH;
                    int idx_kw = kw;
                    int idx = idx_f + idx_c + idx_kh + idx_kw;

                    weight_buf[f][c][kh][kw] = weights[idx];
                }
            }
        }
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        #pragma HLS PIPELINE II=1
        if (tl == 0)
            bias_buf[f] = bias[kernel_offset + f];
        else
            bias_buf[f] = 0;
    }

}

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[],
    const fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_dims_s out_fm_dims,
    const int  ti,
    const int  tj,
    const int  tk
)
{

    const int OUT_FM_DEPTH = out_fm_dims.depth;
    const int OUT_FM_HEIGHT = out_fm_dims.height;
    const int OUT_FM_WIDTH = out_fm_dims.width;

    const dim_t depth_offset  = tk * OUT_BUF_DEPTH;
    const dim_t height_offset = ti * OUT_BUF_HEIGHT;
    const dim_t width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                #pragma HLS PIPELINE II=1
                int idx = (width_offset + j) + (height_offset + i)*OUT_FM_WIDTH + (depth_offset + f)*OUT_FM_WIDTH*OUT_FM_HEIGHT;

                fm_t out;
                // ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out = (fm_t) 0;
                }
                else
                {
                    out = out_fm_buf[f][i][j];
                }

                out_fm[idx] = out;
            }
        }
    }
}
