# ResNet18 CAM on FPGA

This is CAM implementation based on the ResNet18 backbone, implemented entirely for FPGA.
The FPGA board under consideration when writing this HLS implementatin was PYNQ-Z2.

## Code structure
- `src_hls`: Contains the HLS code
    - `src_hls/resnet18.cpp` is the top level module for HLS synthesis
    - `src_hls/*/` contain implementations of different layers, along with vitis synthesis script to check HLS synthesis of individual layers
- `expected_activations` directory contains binary files an example input and expected intermediate activation values
- `src_py/check_mse.py` is the python script to call for checking C++ outputs for accuracy
- `src_py/get_exp_acts.py` generates the expected activation values

## Running a C++ simulation:
1. Go to `src_hls/` and run
```
make sim
./a.out
```

2. To check outputs, go back to project root and run 
```
python src_py/check_mse.py
```

## HLS Synthesis

To start Vitis HLS synthesis, go to `src_hls/` directory
```
make synth
```
