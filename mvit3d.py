import torch
from mobilevit_3d import mobilevit_xxs

img = torch.randn(1, 3, 30, 256, 256)
vit = mobilevit_xxs()
out = vit(img)
