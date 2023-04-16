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
);