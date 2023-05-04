#pragma once

#include "util.h"

/* GAP layer template:
 * 
 * Template parameters:
 *  ID: Input depth
 *  IH: Input height
 *  IW: Input width
*/
template<int ID, int IH, int IW>
void avg_pool(
    fm_t in[ID][IH][IW],
    fm_t out[ID]
)
{
    #pragma HLS inline off

    const fm_t factor = (fm_t) (16.0 / (IH * IW));

    // Zero outputs
    for (int c = 0; c < ID; c++)
        out[c] = 0;

    for (int c = 0; c < ID; c++){
        for (int i = 0; i < IH; i++){
            for (int j = 0; j < IW; j++){
                out[c] += in[c][i][j] / 16;
            }
        }
    }

    for (int c = 0; c < ID; c++){
        out[c] = out[c] * factor;
    }
}
