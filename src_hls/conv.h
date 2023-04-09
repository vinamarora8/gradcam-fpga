#pragma once

#include "util.h"

template <int ID, int IH, int IW, 
          int KD, int KH, int KW, 
          int ST, int PD,
          bool BIAS>
void conv (
        fm_t y[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)],
        fm_t x[ID][IH][IW],
        wt_t weights[KD][ID][KH][KW],
        wt_t biases[KD]
        );
