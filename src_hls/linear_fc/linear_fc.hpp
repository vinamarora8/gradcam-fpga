#pragma once

#include "../util.h"

namespace linear_fc {

void linear_fc(
    fm_t in[512],
    fm_t out[1000],
    wt_t weights[1000][512],
    wt_t biases[1000]
);

}
