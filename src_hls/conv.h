#pragma once

#include "util.h"
#include <cassert>

#define CONV_DIM(IDIM, KDIM, PD, ST) ((IDIM - KDIM + 2*PD)/ST + 1)
// Compute the output dimension given the input dimension, kernel dimension,
// stride and padding

template <int ID, int IH, int IW, 
          int OD, int OH, int OW,
          int KH, int KW, 
          int ST, int PD,
          bool BIAS, bool RES>
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
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                for (int id = 0; id < ID; id++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                        {
                            if (id == 0 && kh == 0 && kw == 0)
                            {
                                if (BIAS)
                                    y[of][oh][ow] = biases[of];
                                else
                                    y[of][oh][ow] = (fm_t) 0;

                                if (RES)
                                    y[of][oh][ow] += res[of][oh][ow];
                            }

                            int i = (ST * oh) - PD + kh;
                            int j = (ST * ow) - PD + kw;

                            if (i < 0 || j < 0 || i >= IH || j >= IW)
                                continue;

                            y[of][oh][ow] += x[id][i][j] * weights[of][id][kh][kw];
                        }
}
