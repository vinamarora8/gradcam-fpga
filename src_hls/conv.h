#pragma once

#include "util.h"

#define CONV_DIM(IDIM, KDIM, PD, ST) ((IDIM - KDIM + 2*PD)/ST + 1)
// Compute the output dimension given the input dimension, kernel dimension,
// stride and padding

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
