#include "../util.h"
#include "linear_fc.hpp"

namespace linear_fc {

template<int NI, int NO>
void linear_fc_core(
    fm_t in[NI],
    fm_t out[NO],
    wt_t weights[NO][NI]
)
{
    #pragma HLS inline off

    #pragma HLS array_partition variable=out complete
    #pragma HLS array_partition variable=weights complete dim=1


    FC_INPUT:
    for (int i = 0; i < NI; i++){
        #pragma HLS pipeline II=1
    FC_OUTPUT:
        for (int j = 0; j < NO; j++){
            #pragma HLS unroll
            out[j] += weights[j][i] * in[i];
        }
    }
}

const int IN_FM_DEPTH = 512;
const int OUT_FM_DEPTH = 1024;

void linear_fc(
    fm_t in[IN_FM_DEPTH],
    fm_t out[OUT_FM_DEPTH],
    wt_t weights[OUT_FM_DEPTH][IN_FM_DEPTH],
    wt_t biases[OUT_FM_DEPTH]
)
{
    #pragma HLS inline off
    #pragma HLS INTERFACE m_axi depth=1  port=in  bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=weights  bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=biases  bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=out bundle=fm_out
    #pragma HLS INTERFACE s_axilite register	port=return

    const int IN_BUF_DEPTH = 512;
    const int OUT_BUF_DEPTH = 16;

    static_assert(OUT_FM_DEPTH % OUT_BUF_DEPTH == 0, "OUT_FM_DEPTH % OUT_BUF_DEPTH != 0");
    static_assert(IN_FM_DEPTH == IN_BUF_DEPTH, "IN_FM_DEPTH != IN_BUF_DEPTH");

    const int N_OP_TILES = OUT_FM_DEPTH / OUT_BUF_DEPTH;

    static fm_t in_buf[IN_BUF_DEPTH];
    static fm_t out_buf[OUT_BUF_DEPTH];
    static fm_t wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH];
    static fm_t bias_buf[OUT_BUF_DEPTH];

    // Load input
    for (int i = 0; i < IN_BUF_DEPTH; i++)
    {
        #pragma HLS pipeline II=1
        in_buf[i] = in[i];
    }

    for (int t = 0; t < N_OP_TILES; t++)
    {
        // Load weights
        for (int i = 0; i < OUT_BUF_DEPTH; i++)
        {
            for (int j = 0; j < IN_BUF_DEPTH; j++)
            {
                #pragma HLS pipeline II=1
                wt_buf[i][j] = weights[t*OUT_BUF_DEPTH + i][j];
            }
        }

        // Load biases into out_buf
        for (int i = 0; i < OUT_BUF_DEPTH; i++)
        {
            #pragma HLS pipeline II=1
            out_buf[i] = biases[t*OUT_BUF_DEPTH + i];
        }


        // Compute
        linear_fc::linear_fc_core<IN_BUF_DEPTH, OUT_BUF_DEPTH>(in_buf, out_buf, wt_buf);


        // Write output
        for (int i = 0; i < OUT_BUF_DEPTH; i++)
        {
            #pragma HLS pipeline II=1
            out[t*OUT_BUF_DEPTH + i] = out_buf[i];
        }
    }
}

}
