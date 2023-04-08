#include "util.h"

template <int IN_BUF_DEPTH, int IN_BUF_HEIGHT, int IN_BUF_WIDTH,
         int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH,
    int KERNEL_HEIGHT, int KERNEL_WIDTH, int STRIDE>
void conv (
        fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
        fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
        wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
        wt_t B_buf[OUT_BUF_DEPTH]
        )
{

    const int S = STRIDE;

    for (int of = 0; of < OUT_BUF_DEPTH; of++)
        for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)
            for (int ow = 0; ow < OUT_BUF_WIDTH; ow++)
                for (int id = 0; id < IN_BUF_DEPTH; id++)
                    for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
                        for (int kw = 0; kw < KERNEL_WIDTH; kw++)
                        {
                            if (id == 0 && kh == 0 && kw == 0)
                                Y_buf[of][oh][ow] = B_buf[of];

                            int i = S*oh + kh;
                            int j = S*ow + kw;

                            Y_buf[of][oh][ow] += X_buf[id][i][j] * W_buf[of][id][kh][kw];
                        }

}
