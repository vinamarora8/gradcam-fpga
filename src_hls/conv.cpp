#include "util.h"
#include "conv.h"

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
 *   BIAS: Whether to use bias
 *   RELU: Whether to use ReLU
 * Templated with the following assumptions:
 * - Padding and Stride are same on all sides
 *
 */
template <int ID, int IH, int IW, 
          int KD, int KH, int KW, 
          int ST, int PD,
          bool BIAS, bool RES>
void conv (
        fm_t y[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)],
        fm_t x[ID][IH][IW],
        wt_t weights[KD][ID][KH][KW],
        wt_t biases[KD],
        fm_t res[KD][CONV_DIM(IH, KH, ST, PD)][CONV_DIM(IW, KW, ST, PD)]
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

                            if (i < 0 || j < 0 || j >= IH || j >= IW)
                                continue;

                            y[of][oh][ow] += x[id][i][j] * weights[of][id][kh][kw];
                        }
}
