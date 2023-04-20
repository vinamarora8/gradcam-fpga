import numpy as np
import os

py_out_path  = 'expected_activations/n01739381_vine_snake/'
cpp_out_path = 'src_hls/out/'

file_list = [
    "conv1_out",
    "maxpool_out",
    "l10_c1_out",
    "l10_c2_out",
    "l11_c1_out",
    "l11_c2_out",
    "l2_ds_out",
    "l20_c1_out",
    "l20_c2_out",
    "l21_c1_out",
    "l21_c2_out",
    "l3_ds_out",
    "l30_c1_out",
    "l30_c2_out",
    "l31_c1_out",
    "l31_c2_out",
    "l4_ds_out",
    "l40_c1_out",
    "l40_c2_out",
    "l41_c1_out",
    "l41_c2_out",
    "avgpool_out",
    "output" 
]

for file_name  in file_list:
    cpp_output = np.fromfile(os.path.join(cpp_out_path, file_name), dtype=float)
    if file_name == 'output':
        file_name == 'fc_out'
    py_output  = np.fromfile(os.path.join(py_out_path,  file_name), dtype=float)

    mse = (np.square(py_output - cpp_output)).mean()
    if mse > 10e-13:
        print("Higher MSE than allowed for {file_name}")
        exit(-1)
