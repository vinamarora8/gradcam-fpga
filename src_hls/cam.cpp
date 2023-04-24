#include "util.h"
#include "max_idx.h"

#define WRITE_TO_FILE_ENABLED

#ifdef WRITE_TO_FILE_ENABLED
#define WRITE_TO_FILE(var, dim0, dim1, dim2) \
    { \
        std::string root_dir = "out/"; \
        std::vector<int> dims(3); \
        dims[0] = dim0; \
        dims[1] = dim1; \
        dims[2] = dim2; \
        write_to_file(root_dir + VAR_NAME(var) + ".bin", dims, var); \
    }
#else
#define WRITE_TO_FILE(var, dim0, dim1, dim2)
#endif

// TODO: I still need to template the dimensions used here
void cam(
        fm_t l41_c2_out[512][7][7],
        wt_t fc_weight[1000][512],
        fm_t output[1000]
        )
{
    fm_t cam_output[7][7];

    WRITE_TO_FILE(fc_weight, 1000, 512, 1);
    WRITE_TO_FILE(output, 1000, 1, 1);

    // Find winning class
    int c = max_idx<1000, fm_t>(output);

    // Weights are 1 x 512
    // Output is 512 x 7 x 7
    for (int k = 0; k < 512; k++){
        l41_c2_out[k][7][7] *= fc_weight[c][k];
    } 

    // Now wk is 512 x 7 x 7
    for (int i = 0; i < 7; i++){
        for (int j = 0; j < 7; j++){
            fm_t temp = 0;

            for (int k = 0; k < 512; k++){
                temp += l41_c2_out[k][i][j]; 
            }

            if (temp > 0)
                cam_output[i][j] = temp/512;
            else
                cam_output[i][j] = 0;
        }
    }

    WRITE_TO_FILE(cam_output, 7, 7, 1);
}