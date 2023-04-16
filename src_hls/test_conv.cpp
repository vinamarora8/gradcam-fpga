// g++ test_conv.cpp

#define CSIM_DEBUG

#include "conv.h"

int main()
{
    {
        float inp[1][3][3] = {
            {
                { 1,  2,  3},
                {11, 12, 13},
                {21, 22, 23}
            }
        };
        float weights[1][1][2][2] = {{{{1, 1}, {1, 1}}}};
        float y[1][2][2];
        conv</*ID*/ 1, /*IH*/ 3, /*IW*/ 3,
            /*OD*/ 1, /*OH*/ 2, /*OW*/ 2,
            /*KH*/ 2, /*KW*/ 2,
            /*ST*/ 2, /*PD*/ 1,
            /*BIAS*/ false, /*RES*/ false> (y, inp, weights, nullptr, nullptr);

        assert(y[0][0][0] == 1);
        assert(y[0][0][1] == 5);
        assert(y[0][1][0] == 32);
        assert(y[0][1][1] == 70);

        /*
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
                std::cout << y[0][i][j] << ", ";

            std::cout << std::endl;
        }
        */
    }
    {
        float inp[1][3][3] = {
            {
                { 1,  2,  3},
                {11, 12, 13},
                {21, 22, 23}
            }
        };
        float weights[1][1][2][2] = {{{{1, 1}, {1, 1}}}};
        float y[1][4][4];
        conv</*ID*/ 1, /*IH*/ 3, /*IW*/ 3,
            /*OD*/ 1, /*OH*/ 4, /*OW*/ 4,
            /*KH*/ 2, /*KW*/ 2,
            /*ST*/ 1, /*PD*/ 1,
            /*BIAS*/ false, /*RES*/ false> (y, inp, weights, nullptr, nullptr);

        float y_exp[1][4][4] = {
            {
                {1, 3, 5, 3},
                {12, 26, 30, 16},
                {32, 66, 70, 36},
                {21, 43, 45, 23}
            }
        };

        for (int i =0; i < 1; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    assert(y_exp[i][j][k] == y[i][j][k]);

        /*
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
                std::cout << y[0][i][j] << ", ";

            std::cout << std::endl;
        }
        */
    }

    std::cout << "CONV TEST PASSED" << std::endl;
}

