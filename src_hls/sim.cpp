#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#define CSIM_DEBUG

#include "util.h"
#include "conv.h"

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

int main(int argc, char *argv[])
{

    return 0;
}
