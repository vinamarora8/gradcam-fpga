#pragma once

#include "util.h"

#define CONV_DIM(IDIM, KDIM, PD, ST) ((IDIM - KDIM + 2*PD)/ST + 1)

template <int ID, int IH, int IW,
    int KD, int KH, int KW,
    int ST, int PD>
void maxpool2d(
    fm_t input[KD][IH][IW],
    fm_t output[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)],
    int max_id[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)]
)
{
    const int OH = CONV_DIM(IH, KH, ST, PD)
    const int OW = CONV_DIM(IW, KW, ST, PD)
    for (int c = 0; c < KD; c++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                fm_t max_val = -1e-09;
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
                max_id[c][h][w] = max_idx;
            }
        }
    }
}

