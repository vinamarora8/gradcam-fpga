import numpy as np
import os

py_out_path  = 'expected_activations/n01739381_vine_snake/'
cpp_out_path = 'src_hls/out/'

file_list = [
    "input.bin",
    "conv1_out.bin",
    "maxpool_out.bin",
    "l10_c1_out.bin",
    "l10_c2_out.bin",
    "l11_c1_out.bin",
    "l11_c2_out.bin",
    "l2_ds_out.bin",
    "l20_c1_out.bin",
    "l20_c2_out.bin",
    "l21_c1_out.bin",
    "l21_c2_out.bin",
    "l3_ds_out.bin",
    "l30_c1_out.bin",
    "l30_c2_out.bin",
    "l31_c1_out.bin",
    "l31_c2_out.bin",
    "l4_ds_out.bin",
    "l40_c1_out.bin",
    "l40_c2_out.bin",
    "l41_c1_out.bin",
    "l41_c2_out.bin",
    "avgpool_out.bin",
    "output.bin" 
]

for file_name  in file_list:
    cpp_output = np.fromfile(os.path.join(cpp_out_path, file_name), dtype=float)
    py_output  = np.fromfile(os.path.join(py_out_path,  file_name), dtype=float)

    mse = (np.square(py_output - cpp_output)).mean()
    print(f"Checking {file_name}")
    print(f"MSE: {mse}")
    print(py_output[:10])
    print(cpp_output[:10])
    print()
