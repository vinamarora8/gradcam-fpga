#include "util.h"


void resize(
    fm_t resizedHeatmap[224][224],
    fm_t cam_output[7][7]
)
{
    // Define the target size
    fm_t targetWidth = 224;
    fm_t targetHeight = 224;
    printf("here");
    // Create a new heatmap array with the target size
    //fm_t resizedHeatmap[targetHeight][targetWidth];

    // Calculate the scaling factors for width and height
    fm_t scaleX = 7.0/224;
    fm_t scaleY = 7.0/224;
    printf("scaleX is %2f", scaleX);
    printf("scaleY is %2f", scaleY);
    // Iterate over the target size and interpolate the heatmap values
    for (int y = 0; y < targetHeight; y++) {
        for (int x = 0; x < targetWidth; x++) {
            // Calculate the corresponding position in the original heatmap
            fm_t origX = x * scaleX;
            fm_t origY = y * scaleY;

            // Calculate the integer indices of the surrounding pixels
            int x0 = (int)origX;
            int y0 = (int)origY;
            int x1 = std::min(x0 + 1, 6);
            int y1 = std::min(y0 + 1, 6);

            // Calculate the fractional parts for interpolation
            fm_t dx = origX - x0;
            fm_t dy = origY - y0;
            fm_t value=0.0;
            
            
            // Interpolate the heatmap values using bilinear interpolation
            value = (1 - dx) * (1 - dy) * cam_output[y0][x0] + dx * (1 - dy) * cam_output[y0][x1] + (1 - dx) * dy * cam_output[y1][x0] + dx * dy * cam_output[y1][x1];
            
            // Assign the interpolated value to the resized heatmap
            resizedHeatmap[y][x] = value;
        }
    }
}