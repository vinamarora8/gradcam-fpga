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
    fm_t in[IH][IH][IW],
    fm_t out[IH]
)
{
    for (int c = 0; c < ID; c++){
#pragma HLS pipeline off
        fm_t sum = 0;

        for (int i = 0; i < IH; i++){
#pragma HLS pipeline off
            for (int j = 0; j < IW; j++){
#pragma HLS pipeline off
#pragma HLS unroll factor=1
                sum += in[c][i][j];
            }
        }

        out[c] = sum / (IH * IW);
    }
}
