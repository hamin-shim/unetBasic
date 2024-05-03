import torch
from mobilevit_3d_light import MobileViT
import torch.nn as nn
import os

# Arrange GPU devices starting from 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPUs 2 and 3 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
img_size = (50, 128, 128)
img = torch.randn(1, 3, *img_size).to(device)
dims = [16, 32]
# channels = [8, 16, 32, 64, 128]
channels = [16, 32, 64, 128, 128]

_model = MobileViT(img_size, dims, channels,
                   expansion=2).to(device)
# model = nn.DataParallel(_model).to(device)
out = _model(img)
