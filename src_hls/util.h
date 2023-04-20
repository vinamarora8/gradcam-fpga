#pragma once

#include <iostream>

#ifdef CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
#include <ap_fixed.h>
    typedef ap_fixed<16,3> fm_t;
    typedef ap_fixed<16,3> wt_t;
#endif

#define VAR_NAME(x) #x

float *read_bin_file(const std::string filepath, const std::vector<int> dims)
{
    uint64_t size = 1;
    for (int i = 0; i < dims.size(); i++)
        size *= dims[i];

    float *ans;
    ans = (float *) malloc(size * sizeof(float));

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*)(ans), size*sizeof(float));
    input.close();

    return ans;
}

// template <typename fixp_t, int F, int C, int M, int N>
// void load_fp_and_fixp_vals(float**** fp_arr, fixp_t**** fixp_arr, const std::string filepath){
// 
//     std::ifstream input(filepath, std::ios::in | std::ios::binary);
//     input.read((char*) (***fp_arr), F*C*M*N*sizeof(float));
//     input.close();
// 
//     fixp_arr = (fixp_t*) fp_arr;
// }

    // if (dims.size() == 4) {
    //     for (int f = 0; f < dims[0]; f++)
    //         for (int c = 0; c < dims[1]; c++)
    //             for (int m = 0; m < dims[2]; m++)
    //                 for (int n = 0; n < dims[3]; n++){
    //                     fp_arr[f][c][m][n] = bin_vals[f][c][m][n];
    //                     fixp_arr[f][c][m][n] = (fm_t) fp_arr[f][c][m][n];
    //                 }
    // } else if (dims.size() == 3){
    //     for(int c = 0; c < 3; c++)
    //         for(int i = 0; i < 736; i++)
    //             for(int j = 0; j < 1280; j++){
    //                 fp_arr[c][i][j] = bin_vals[c][i][j];
    //                 fixp_arr[c][i][j] = (fm_t) fp_arr[c][i][j];
    //             }
    // } else if (dims.size() == 2){
    //     for(int i = 0; i < 736; i++)
    //         for(int j = 0; j < 1280; j++){
    //             fp_arr[i][j] = bin_vals[i][j];
    //             fixp_arr[i][j] = (fm_t) fp_arr[i][j];
    //         }
    // } else if (dims.size() == 1){
    //     for(int j = 0; j < 1280; j++){
    //         fp_arr[j] = bin_vals[j];
    //         fixp_arr[j] = (fm_t) fp_arr[j];
    //     }
    // }


// void load_fp_and_fixp_vals(float* fp_arr, wt_t* fixp_arr, 
//     const std::string filepath, const std::vector<int> dims){
// 
//     fp_arr = read_bin_file(filepath, dims);
//     fixp_arr = (wt_t*) fp_arr;
// 
//     // if (dims.size() == 4) {
//     //     for (int f = 0; f < dims[0]; f++)
//     //         for (int c = 0; c < dims[1]; c++)
//     //             for (int m = 0; m < dims[2]; m++)
//     //                 for (int n = 0; n < dims[3]; n++){
//     //                     fp_arr[f][c][m][n] = bin_vals[f][c][m][n];
//     //                     fixp_arr[f][c][m][n] = (wt_t) fp_arr[f][c][m][n];
//     //                 }
//     // } else if (dims.size() == 3){
//     //     for(int c = 0; c < 3; c++)
//     //         for(int i = 0; i < 736; i++)
//     //             for(int j = 0; j < 1280; j++){
//     //                 fp_arr[c][i][j] = bin_vals[c][i][j];
//     //                 fixp_arr[c][i][j] = (wt_t) fp_arr[c][i][j];
//     //             }
//     // } else if (dims.size() == 2){
//     //     for(int i = 0; i < 736; i++)
//     //         for(int j = 0; j < 1280; j++){
//     //             fp_arr[i][j] = bin_vals[i][j];
//     //             fixp_arr[i][j] = (wt_t) fp_arr[i][j];
//     //         }
//     // } else if (dims.size() == 1){
//     //     for(int j = 0; j < 1280; j++){
//     //         fp_arr[j] = bin_vals[j];
//     //         fixp_arr[j] = (wt_t) fp_arr[j];
//     //     }
//     // }
// }

template <typename fixp_t>
void write_to_file(const std::string& filename, const std::vector<int>& dims, const fixp_t* array) {
  size_t num_elements = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    num_elements *= dims[i];
  }

  std::cout << "Writing to file: " << filename << std::endl;
  const float* float_array = reinterpret_cast<const float*>(array);

  std::ofstream output_file(filename, std::ios::out | std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(float_array), num_elements * sizeof(float));

  output_file.close();
}

// void write_to_file(const std::string& filename, const std::vector<int>& dims, const wt_t* array) {
//   size_t num_elements = 1;
//   for (size_t i = 0; i < dims.size(); i++) {
//     num_elements *= dims[i];
//   }
// 
//   const float* float_array = reinterpret_cast<const float*>(array);
// 
//   std::ofstream output_file(filename, std::ios::out | std::ios::binary);
//   output_file.write(reinterpret_cast<const char*>(float_array), num_elements * sizeof(float));
// 
//   output_file.close();
// }

template <typename fixp_t, int F, int C, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[F][C][M][N], fixp_t fixp_arr[F][C][M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*) (fp_arr), F*C*M*N*sizeof(float));
    input.close();

    for (int f = 0; f < F; f++) {
        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    fixp_arr[f][c][m][n] = static_cast<fm_t>(fp_arr[f][c][m][n]);
                }
            }
        }
    }
}

template <typename fixp_t, int C, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[C][M][N], fixp_t fixp_arr[C][M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*) (fp_arr), C*M*N*sizeof(float));
    input.close();

    for (int c = 0; c < C; c++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                fixp_arr[c][m][n] = static_cast<fm_t>(fp_arr[c][m][n]);
            }
        }
    }
}

template <typename fixp_t, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[M][N], fixp_t fixp_arr[M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*) (fp_arr), M*N*sizeof(float));
    input.close();

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            fixp_arr[m][n] = static_cast<fm_t>(fp_arr[m][n]);
        }
    }
}

template <typename fixp_t, int N>
void load_fp_and_fixp_vals(float fp_arr[N], fixp_t fixp_arr[N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*) (fp_arr), N*sizeof(float));
    input.close();

    for (int n = 0; n < N; n++) {
        fixp_arr[n] = static_cast<fm_t>(fp_arr[n]);
    }
}
