import torch
from model import UNet3d

model_name = 'model_revised_interval_3'
# Loading the serialized model to avoid computation
model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
model.load_state_dict(torch.load(f'saved_model/best-{model_name}.pth'))

# Turning on Evaluation mode of the model
model.eval()
