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
    //int max_id[OD][OH][OW]
)
{
    assert(OH == CONV_DIM(IH, KH, PD, ST));
    assert(OW == CONV_DIM(IW, KW, PD, ST));

    for (int c = 0; c < OD; c++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                fm_t max_val = 0;
                int max_idx = 0;
                for (int i = 0; i < KH; i++) {
                    for (int j = 0; j < KW; j++) {
                        int row_idx = i*ST + KH - PD;
                        int col_idx = j*ST + KW - PD;
                        if (row_idx >= 0 && col_idx >=  0 && row_idx < IH && col_idx < IW) 
                        {
                            fm_t val = input[c][row_idx][col_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = KW * i + j;
                            }
                        }
                        
                    }
                }
                output[c][h][w] = max_val;
                //max_id[c][h][w] = max_idx;
            }
        }
    }
}

