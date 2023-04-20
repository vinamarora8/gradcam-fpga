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
    for (int j = 0; j < NO; j++){
        fm_t sum = biases[j];

        for (int i = 0; i < NI; i++){
            sum += weights[j][i] * in[i];
        }

        out[j] = sum;
    }
}
