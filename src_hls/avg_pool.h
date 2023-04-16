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
        fm_t sum = 0;

        for (int i = 0; i < IH; i++){
            for (int j = 0; j < IW; j++){
                sum += in[c][i][j];
            }
        }

        out[c] = sum / (IH * IW);
    }
}
