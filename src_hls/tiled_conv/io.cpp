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
    #pragma HLS inline
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
        #pragma HLS PIPELINE off
        int idx_h = height_offset - P;
        int idx_hm = FM_HEIGHT*(height_offset - P);
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            #pragma HLS PIPELINE off
            int idx_w = width_offset - P;
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < BUF_WIDTH; j++)
            {
                #pragma HLS PIPELINE off
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


template<int OUT_BUF_DEPTH, int IN_BUF_DEPTH, int KERNEL_HEIGHT, int KERNEL_WIDTH>
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
    #pragma HLS inline
    const int kernel_offset  = tk * OUT_BUF_DEPTH;
    const int tile_layer_offset = tl * IN_BUF_DEPTH;

    int idx_f = kernel_offset*IN_FM_DEPTH*KERNEL_HEIGHT*KERNEL_WIDTH;
    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        #pragma HLS PIPELINE off
        int idx_c = tile_layer_offset*KERNEL_HEIGHT*KERNEL_WIDTH;
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            #pragma HLS PIPELINE off
            int idx_kh = 0;
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < KERNEL_HEIGHT; kh++)
	        {
                #pragma HLS PIPELINE off
                int idx_kw = 0;
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < KERNEL_WIDTH; kw++)
	            {
                    weight_buf[f][c][kh][kw] = weights[idx_f + idx_c + idx_kh + idx_kw];
                    idx_kw++;
                }
                idx_kh += KERNEL_WIDTH;
            }
            idx_c += KERNEL_HEIGHT*KERNEL_WIDTH;
        }
        idx_f += IN_FM_DEPTH*KERNEL_HEIGHT*KERNEL_WIDTH;
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = (tl == 0) ? bias[kernel_offset + f] : (wt_t) 0;
    }

}

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[],
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const int  ti,
    const int  tj,
    const int  tk,
    const bool relu,
    const int  W,
    const int  WxH
)
{
    #pragma HLS inline off

    int idx_d = tk * OUT_BUF_DEPTH;
    int idx_h = ti * OUT_BUF_HEIGHT;
    int idx_w = tj * OUT_BUF_WIDTH;

    int idx = idx_d*WxH + idx_h*W + idx_w;

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

                //int idx = (idx_w + j) + (idx_h + i)*W + (idx_d + f)*WxH; 

                // ReLU in-place
                /*
                if(relu & (out_fm_buf[f][i][j] < (fm_t) 0))
                {
                    out_fm_buf[f][i][j] = (fm_t) 0;
                }
                */
                
                out_fm[idx] = out_fm_buf[f][i][j];

                idx++;

                if (j == OUT_BUF_WIDTH-1)
                {
                    idx += W - OUT_BUF_WIDTH;

                    if (i == (OUT_BUF_HEIGHT-1))
                    {
                        idx += WxH - W*OUT_BUF_HEIGHT;
                    }
                }
             }
        }
    }
}
