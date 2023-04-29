#include "util.h"

template<int D, int H, int W>
void add_residue(
    fm_t fm_in[D][H][W],
    const fm_t res[D][H][W]
)
{
    for (int i = 0; i < D; i++)
        for (int j = 0; j < H; j++)
            for (int k = 0; k < W; k++)
            {
                fm_t x = fm_in[i][j][k] + res[i][j][k];
                if (x < 0)
                {
                    x = (fm_t) 0;
                }
                
                fm_in[i][j][k] = x;
            }

}
