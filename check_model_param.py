import torch
from mobilevit import mobilevit_xxs, count_parameters

img = torch.randn(1, 3, 256, 256)
vit = mobilevit_xxs()
out = vit(img)
print(count_parameters(vit))
