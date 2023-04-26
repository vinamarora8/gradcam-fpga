from traitlets.config.application import T
from torchvision.models.resnet import resnet152
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights, resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
from pprint import pprint
from google.colab.patches import cv2_imshow
from IPython.display import Image
import cv2

is18 = not False
is152 = not is18

if (is18):
    weights = ResNet18_Weights.IMAGENET1K_V1
else:
    weights = ResNet152_Weights.IMAGENET1K_V1

#file_path = "/content/sample_data/n02028035_redshank.jpeg"
# file_path = "/content/sample_data/n04146614_school_bus.jpeg"
# file_path = "/content/sample_data/n04141975_scale.jpeg"
# file_path = "/content/sample_data/cat_heatmap.png"
file_path = "/content/sample_data/n01739381_vine_snake.jpeg"

# ResNet Class
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        # define the resnet152
        if (is18):
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet152(weights=weights)
        
        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      # self.resnet.relu,
                                      # self.resnet.maxpool,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # average pooling layer
        self.avgpool = self.resnet.avgpool
        
        # classifier
        self.classifier = self.resnet.fc
        
        # gradient placeholder
        self.gradient = None
    
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x

      
# all the data transformation and loading
# transform = transforms.Compose([transforms.Resize((224, 224)),
#                                transforms.ToTensor(), 
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#dataset = ImageFolder(root='/content/sample_data', transform=transform)
#dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

preprocess = weights.transforms()
i = read_image(file_path)
img = preprocess(i).unsqueeze(0)

# init the resnet
resnet = ResNet()

# set the evaluation mode
_ = resnet.eval()

# get the image
#img, _ = next(iter(dataloader))

# forward pass
pred = resnet(img)


prediction = pred.squeeze(0).softmax(0)
class_id = prediction.argmax().item()
print(f"Class ID is {class_id}")
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
pprint(f"{category_name}: {100 * score:.1f}%")

#pred.argmax(dim=1) # prints tensor([2])

# get the gradient of the output with respect to the parameters of the model
pred[:, 2].backward()

# pull the gradients out of the model
gradients = resnet.get_gradient()
print("Gradients size is " + str(gradients.size()))

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
print("Pooled gradients size is " + str(pooled_gradients.size()))
print(pooled_gradients[:10])

debug_w = resnet.resnet.fc.weight
print("FC wts size is " + str(debug_w.size()))
print(debug_w[class_id])

# get the activations of the last convolutional layer
activations = resnet.get_activations(img).detach()
print("Activations size is " + str(activations.size()))

# weight the channels by corresponding gradients
# Last conv layer has 512 channels
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

print("Post mult activations size is " + str(activations.size()))
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()
print("Heatmap size is " + str(heatmap.size()))
print(heatmap)

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

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

Image(filename='map.jpg') 
