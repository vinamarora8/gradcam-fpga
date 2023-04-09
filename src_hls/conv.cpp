#include "util.h"

#define CONV_DIM(IDIM, KDIM, PD, ST) ((IDIM - KDIM + 2*PD)/ST + 1)
// Compute the output dimension given the input dimension, kernel dimension,
// stride and padding

/* Convolution kernel template
 *
 * Template parameters:
 *   ID: Input depth
 *   IH: Input height
 *   IW: Input width
 *   KD: Kernel depth
 *   KH: Kernel height
 *   KW: Kernel width
 *   ST: Stride
 *   PD: Padding
 * Templated with the following assumptions:
 * - Padding and Stride are same on all sides
 *
 */
template <int ID, int IH, int IW, int KD, int KH, int KW, int ST, int PD>
void conv (
        fm_t Y_buf[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)],
        fm_t X_buf[ID][IH][IW],
        wt_t W_buf[KD][ID][KH][KW],
        wt_t B_buf[KD]
        )
{

    const int OD = KD;
    const int OH = CONV_DIM(IH, KH, ST, PD);
    const int OW = CONV_DIM(IW, KW, ST, PD);

    for (int of = 0; of < KD; of++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                for (int id = 0; id < ID; id++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                        {
                            if (id == 0 && kh == 0 && kw == 0)
                                Y_buf[of][oh][ow] = B_buf[of];

                            int i = (ST * oh) - PD + kh;
                            int j = (ST * ow) - PD + kw;

                            if (i < 0 || j < 0 || j >= IH || j >= IW)
                                continue;

                            Y_buf[of][oh][ow] += X_buf[id][i][j] * W_buf[of][id][kh][kw];
                        }

}
