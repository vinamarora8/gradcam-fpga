#include "util.h"
#include "max_idx.h"



// TODO: I still need to template the dimensions used here
void cam(
        fm_t cam_output[7][7],
        fm_t l4_out1[1][512][7][7],
        wt_t fc_weight[1000][512],
        fm_t output[1000]
        )
{
    //fm_t cam_output[7][7];

//    WRITE_TO_FILE(fc_weight, 1000, 512, 1);
//    WRITE_TO_FILE(output, 1000, 1, 1);

    // Find winning class
    int c = max_idx<1000, fm_t>(output);
    
    for (int m=0; m<1; m++){
        for (int k = 0; k < 512; k++){
            for (int i=0; i<7; i++){
                for (int j=0; j<7; j++){
                    l4_out1[m][k][i][j] *= fc_weight[59][k];
                }
            }
        } 
    }
    // Now wk is 512 x 7 x 7
fm_t maxVal = cam_output[0][0];
for (int i = 0; i < 7; i++){
    for (int j = 0; j < 7; j++){
        fm_t temp = 0;

        for (int k = 0; k < 512; k++)
        {
            temp += l4_out1[0][k][i][j];
        }
        
        //RELU
        if (temp<0){
            cam_output[i][j] =0;
        }
        else{
            
            cam_output[i][j]= temp/(512*7*7); 
        }
        if(cam_output[i][j]>maxVal){
            maxVal= cam_output[i][j];
        }
        
    }
}

printf("max value is %2f", maxVal);
for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
        cam_output[i][j] = cam_output[i][j]/ maxVal;
    }
}
    //WRITE_TO_FILE(cam_output, 7, 7, 1);
}
