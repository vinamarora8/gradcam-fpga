import numpy as np
import os
import torch
import cv2
# import matplotlib.pyplot as plt

py_out_path  = 'expected_activations/n01739381_vine_snake/'
cpp_out_path = 'src_hls/out/'

cam_out = np.fromfile(os.path.join(cpp_out_path, 'cam_output.bin'), dtype=np.float32)
weights = np.fromfile(os.path.join(cpp_out_path, 'fc_weight.bin'), dtype=np.float32)
output = np.fromfile(os.path.join(cpp_out_path, 'output.bin'), dtype=np.float32)

heatmap = cam_out 

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())

# make the heatmap to be a numpy array
heatmap = heatmap.numpy()

# interpolate the heatmap
img = cv2.imread(file_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)