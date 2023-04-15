#include <stdio.h>
#include <stdlib.h>
#include <util.h>
#include <conv.h>

/*Parameters
*    NUM_CHANNELS: i/p Depth
 *   IM_HEIGHT: i/p height
 *   IM_WIDTH: i/p width
 *   OUT_HEIGHT: o/p height
 *   OUT_WIDTH: o/p width
 *   Assumptions
 *   3*3 window(kernel_size) with a stride of 2 padding of 1
 * Keeping i/p dims as 224*224*64 for now.
*/


/* #define IH 224
#define IW 224
#define KD 64
#define KH 3
#define KW 3
#define ST 2
#define PD 1
#define OH IM_HEIGHT/2
#define OW IM_WIDTH/2
*/
template <int ID, int IH, int IW,
          int KD, int KH, int KW,
          int ST, int PD>




void maxpool2d(fm_t input[KD][IH][IW],
    fm_t output[KD][OH][OW],
    int max_id[KD][OH][OW]) 
{
    for (int c = 0; c < KD; c++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                fm_t max_val = -1e-09;
                int max_idx = 0;
                for (int i = 0; i < KH; i++) {
                    for (int j = 0; j < KW; j++) {
                        int row_idx = i*ST + KH - PD;
                        int col_idx = j*ST + KW - PD;
                        if (row_idx >= 0 && col_idx >=  0 && row_idx < IH && col_idx < IW) 
                        {
                            fm_t val = input[c][row_idx][col_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = KW * i + j;
                            }
                        }
                        
                    }
                }
                output[c][h][w] = max_val;
                max_id[c][h][w] = max_idx;
            }
        }
    }
}




