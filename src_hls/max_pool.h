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
);
