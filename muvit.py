import torch
from vitUnet import MUnetViT
import os
import torch.nn as nn

# Arrange GPU devices starting from 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set the GPUs 2 and 3 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

img_size = (64, 128, 128)
img = torch.randn(2, 3, *img_size)
dims = [16, 32]
n_classes = 3
# channels = [8, 16, 32, 64, 128]
channels = [16, 32, 64, 128, 128]

_model = MUnetViT(img_size, dims, channels,
                  n_classes, expansion=2).to(device)
model = nn.DataParallel(_model).to(device)
out = model(img)
print(out.shape)
