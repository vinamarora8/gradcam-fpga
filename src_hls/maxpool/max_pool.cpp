#pragma once

#include "../conv.h"
#include <cassert>
namespace maxpool 
{
 #include "params.hpp"
 //#include "conv.cpp"
 #include "io.cpp"
 #include "max.cpp"
template <int ID, int IH, int IW,
          int OD, int OH, int OW,
          int KH, int KW,
          int ST, int PD>

void maxpool2d(
    fm_t output[64][56][56],
    fm_t input[64][112][112]
)
{
    #pragma HLS INTERFACE m_axi depth=1  port=input   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=output bundle=fm_out
    #pragma HLS inline off
    int tile_height = 14;
    int tile_width=14;

    
    //fm_t in_buf[64][112][112];
    fm_t in_buf[64][tile_height][tile_width];
    fm_t out_buf[64][tile_height][tile_width];
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
        maxpool::load_fm_tile_block_from_DRAM<IN_BUF_DEPTH, TILE_HEIGHT, TILE_WIDTH, PADDING, IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH>
                (in_buf, input, ti, tj, 0);
        
    KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GRPS; tk++)
            {
                maxpool::maxpool2d<64, 112, 112, 64, 14,  14, 3, 3, 2, 1 >(output, in_buf);
        
            
    
        maxpool::store_output_tile_to_DRAM<OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH>
                    (output, out_buf, ti, tj, tk, 0);
    
            }
        }
    }


void test_maxpool(
   fm_t output[64][56][56],
   fm_t input[64][112][112] 
)
{
    maxpool::maxpool2d<64, 112, 112, 64, 56,  56, 3, 3, 2, 1 >(
        (fm_t (*)[56][56]) output ,
        (fm_t (*)[112][112]) input
        );

}
}
}
