#pragma once

#include "util.h"

/* Linear fully-connected layer template:
 * 
 * Template parameters:
 *  NI: Input features
 *  NO: Output features
 *  BIAS: Whether to use bias
*/
template<int NI, int NO>
void linear_fc(
    fm_t in[NI],
    fm_t out[NO],
    wt_t weights[NO][NI],
    wt_t biases[NO]
)
{
FC_OUTPUT:
    for (int j = 0; j < NO; j++){
#pragma HLS pipeline off
        fm_t sum = biases[j];

    FC_INPUT:
        for (int i = 0; i < NI; i++){
#pragma HLS pipeline off
#pragma HLS unroll factor=1
            sum += weights[j][i] * in[i];
        }

        out[j] = sum;
    }
}
