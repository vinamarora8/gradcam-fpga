#pragma once
#include "../util.h"

template<
int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH,
int IN_BUF_DEPTH, int IN_BUF_HEIGHT, int IN_BUF_WIDTH,
int KERNEL_HEIGHT, int KERNEL_WIDTH, int STRIDE>
void conv_small (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    const wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t B_buf[OUT_BUF_DEPTH]
)
{
    const int S = STRIDE;

OUT_FEAT:
    for (int of = 0; of < OUT_BUF_DEPTH; of++)
    OUT_ROW:
        for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)
        OUT_COL:
            for (int ow = 0; ow < OUT_BUF_WIDTH; ow++)
            IN_FEAT:
                for (int id = 0; id < IN_BUF_DEPTH; id++)
              IN_ROW:
                    for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
                    IN_COL:
                        for (int kw = 0; kw < KERNEL_WIDTH; kw++)
                        {
                            if (id == 0 && kh == 0 && kw == 0)
                            {
                                Y_buf[of][oh][ow] = B_buf[of];
                            }

                            int i = S*oh + kh;
                            int j = S*ow + kw;

                            Y_buf[of][oh][ow] += X_buf[id][i][j] * W_buf[of][id][kh][kw];
                        }
}
