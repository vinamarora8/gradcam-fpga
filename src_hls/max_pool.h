#pragma once

#include "conv.h"
#include <cassert>

template <int ID, int IH, int IW,
          int OD, int OH, int OW,
          int KH, int KW,
          int ST, int PD>
void maxpool2d(
    fm_t output[OD][OH][OW],
    fm_t input[ID][IH][IW]
)
{
    assert(OH == CONV_DIM(IH, KH, PD, ST));
    assert(OW == CONV_DIM(IW, KW, PD, ST));

MP_FEAT:
    for (int c = 0; c < OD; c++) {
    #pragma HLS pipeline off
    MP_ROW:
        for (int h = 0; h < OH; h++) {
        #pragma HLS pipeline off
        MP_COL:
            for (int w = 0; w < OW; w++) 
            #pragma HLS pipeline off
            {
                fm_t max_val = -1e30;
                MP_KER_ROW:
                for (int i = 0; i < KH; i++) 
                #pragma HLS pipeline off
                {
                    MP_KER_COL:
                    for (int j = 0; j < KW; j++) 
                    #pragma HLS pipeline off
                    {
                        int row_idx = (ST * h) + i - PD;
                        int col_idx = (ST * w) + j - PD;
                        if (row_idx >= 0 && col_idx >=  0 && row_idx < IH && col_idx < IW) 
                        {
                            fm_t val = input[c][row_idx][col_idx];
                            if (val > max_val) 
                            {
                                max_val = val;
                            }
                        }
                    }
                }

                output[c][h][w] = max_val;
            }
        }
    }
}

