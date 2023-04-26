#pragma once

#include "util.h"
#include <cassert>

#define CONV_DIM(IDIM, KDIM, PD, ST) ((IDIM - KDIM + 2*PD)/ST + 1)
// Compute the output dimension given the input dimension, kernel dimension,
// stride and padding

static fm_t relu(fm_t x)
{
    if (x < 0)
        x = 0;

    return x;
}

template <int ID, int IH, int IW, 
          int OD, int OH, int OW,
          int KH, int KW, 
          int ST, int PD,
          bool BIAS, bool RES, bool RELU=true>
void conv (
        fm_t y[OD][OH][OW],
        fm_t x[ID][IH][IW],
        wt_t weights[OD][ID][KH][KW],
        wt_t biases[OD],
        fm_t res[OD][OH][OW]
        )
{
    assert(OH == CONV_DIM(IH, KH, PD, ST));
    assert(OW == CONV_DIM(IW, KW, PD, ST));

    for (int of = 0; of < OD; of++)
#pragma HLS pipeline off
        for (int oh = 0; oh < OH; oh++)
#pragma HLS pipeline off
            for (int ow = 0; ow < OW; ow++)
            {
#pragma HLS unroll factor=1
#pragma HLS pipeline off
                for (int id = 0; id < ID; id++)
#pragma HLS unroll factor=1
                    for (int kh = 0; kh < KH; kh++)
#pragma HLS unroll factor=1
                        for (int kw = 0; kw < KW; kw++)
                        {
#pragma HLS unroll factor=1
                            if (id == 0 && kh == 0 && kw == 0)
                            {
                                if (RES)
                                    y[of][oh][ow] = res[of][oh][ow];
                                else
                                    y[of][oh][ow] = (fm_t) 0;

                                if (BIAS)
                                    y[of][oh][ow] += biases[of];

                            }

                            int i = (ST * oh) - PD + kh;
                            int j = (ST * ow) - PD + kw;

                            if (i < 0 || j < 0 || i >= IH || j >= IW)
                                continue;

                            y[of][oh][ow] += x[id][i][j] * weights[of][id][kh][kw];
                        }

                if (RELU)
                    y[of][oh][ow] = relu(y[of][oh][ow]);
            }
}
