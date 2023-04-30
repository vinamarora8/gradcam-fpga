#pragma once
#include "../util.h"

template<int TILE_DEPTH, int TILE_HEIGHT, int TILE_WIDTH, int PADDING>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[TILE_DEPTH][TILE_HEIGHT + 2*PADDING][TILE_WIDTH + 2*PADDING],
    const fm_t in_fm[],
    const fm_dims_s dims,
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

    dim_t FM_DEPTH = dims.depth;
    dim_t FM_HEIGHT = dims.height;
    dim_t FM_WIDTH = dims.width;

    int idx_dm = FM_HEIGHT*FM_WIDTH*depth_offset;
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < BUF_DEPTH; c++) // FM and BUF have same depth
    {
        int idx_h = height_offset - P;
        int idx_hm = FM_HEIGHT*(height_offset - P);
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            int idx_w = width_offset - P;
            for(int j = 0; j < BUF_WIDTH; j++)
            {

                if ((idx_h < 0 || idx_h >= FM_HEIGHT) ||
                    (idx_w < 0 || idx_w >= FM_WIDTH))
                {
                    in_fm_buf[c][i][j] = (fm_t) 0;
                }
                else
                {
                    int idx = idx_w + idx_hm + idx_dm;
                    in_fm_buf[c][i][j] = in_fm[idx];
                }
                idx_w++;
            }
            idx_h++;
            idx_hm += FM_HEIGHT;
        }
        idx_dm += FM_WIDTH*FM_HEIGHT;
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

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[],
    const fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_dims_s out_fm_dims,
    const int  ti,
    const int  tj,
    const int  kernel_group,
    const bool relu
)
{
    const int OUT_FM_DEPTH = out_fm_dims.depth;
    const int OUT_FM_HEIGHT = out_fm_dims.height;
    const int OUT_FM_WIDTH = out_fm_dims.width;

    const dim_t depth_offset  = kernel_group * OUT_BUF_DEPTH;
    const dim_t height_offset = ti * OUT_BUF_HEIGHT;
    const dim_t width_offset  = tj * OUT_BUF_WIDTH;

    int idx_d = (depth_offset)*OUT_FM_WIDTH*OUT_FM_HEIGHT;
    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        int idx_h = (height_offset)*OUT_FM_WIDTH;
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            int idx_w = width_offset;
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                int idx = idx_w + idx_h + idx_d;

                // ReLU in-place
                if(relu & (out_fm_buf[f][i][j] < (fm_t) 0))
                {
                    out_fm[idx] = (fm_t) 0;
                }
                else
                {
                    out_fm[idx] = out_fm_buf[f][i][j];
                }
                idx_w++;
            }
            idx_h += OUT_FM_WIDTH;
        }
        idx_d += OUT_FM_WIDTH*OUT_FM_HEIGHT;
    }
}
