#pragma once

#include "../util.h"

/* Linear fully-connected layer template:
 * 
 * Template parameters:
 *  NI: Input features
 *  NO: Output features
 *  BIAS: Whether to use bias
*/
template<int NI, int NO>
void linear_fc(
    fm_t dram_in[NI],
    fm_t dram_out[NO],
    wt_t dram_weights[NO][NI],
    wt_t dram_biases[NO]
)
{
    #pragma HLS inline off

    #pragma HLS INTERFACE m_axi depth=1  port=dram_in       bundle=fm_t
    #pragma HLS INTERFACE m_axi depth=1  port=dram_out      bundle=fm_t
    #pragma HLS INTERFACE m_axi depth=1  port=dram_weights  bundle=wt_t
    #pragma HLS INTERFACE m_axi depth=1  port=dram_biases   bundle=wt_t
    #pragma HLS INTERFACE s_axilite register	port=return

    const int TILE_SIZE = 8;

    // fm_t buf_in[NI];
    // fm_t buf_out[NO];
    // wt_t buf_weights[NO][NI];
    // wt_t buf_biases[NO];

    fm_t buf_in[TILE_SIZE];
    fm_t buf_out[TILE_SIZE];
    wt_t buf_weights[TILE_SIZE][TILE_SIZE];
    wt_t buf_biases[TILE_SIZE];


    for (int ti = 0; ti < NI/TILE_SIZE; ti++){
        for (int tj = 0; tj < NO/TILE_SIZE; tj++){

            for (int t = 0; t < TILE_SIZE; t++){

                // Load
                buf_in[t] = dram_in[t + ti * TILE_SIZE];
                buf_biases[t] = dram_biases[t + tj * TILE_SIZE];

                for (int u = 0; u < TILE_SIZE; u++){
                    buf_weights[t][u] = dram_weights[t + tj * TILE_SIZE][u + ti * TILE_SIZE];
                }
            }

            // Compute
            FC_OUTPUT:
            for (int j = 0; j < TILE_SIZE; j++){
                #pragma HLS pipeline off 
                fm_t sum = buf_biases[j];

                FC_INPUT:
                for (int i = 0; i < TILE_SIZE; i++){
                    #pragma HLS pipeline off
                    sum += buf_weights[j][i] * buf_in[i];
                }

                buf_out[j] = sum;
            }

            // Write out
            for (int i = 0; i < TILE_SIZE; i++) {
                dram_out[i + tj * TILE_SIZE] = buf_out[i];
            }

        }
    }
    


    // for (int i = 0; i < NI; i++){
    //     buf_in[i] = dram_in[i];
    // } 

    // for (int i = 0; i < NO; i++){
    //     for (int j = 0; j < NI; j++){
    //         buf_weights[i][j] = dram_weights[i][j];
    //     }
    // } 

    // for (int i = 0; i < NO; i++){
    //     buf_biases[i] = dram_biases[i];
    // } 

// FC_OUTPUT:
//     for (int j = 0; j < NO; j++){
// #pragma HLS pipeline off
//         fm_t sum = buf_biases[j];

//     FC_INPUT:
//         for (int i = 0; i < NI; i++){
// #pragma HLS pipeline off
//             sum += buf_weights[j][i] * buf_in[i];
//         }

//         buf_out[j] = sum;
//     }

    // for (int i = 0; i < NO; i++){
    //     dram_out[i] = buf_out[i];
    // } 
}

// void test_linear_fc(
//     fm_t dummy_in[512],
//     fm_t dummy_out[1000],
//     wt_t dummy_weights[1000][512],
//     wt_t dummy_biases[1000]
// )
// {
//     linear_fc<512, 1000>((fm_t*) dummy_in, (fm_t*) dummy_out, (wt_t (*)[512]) dummy_weights, (wt_t*) dummy_biases);
// }
