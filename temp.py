import torch
from model import UNet3d

model_name = 'int3_inp3_nch32_120'
save_path = f'models/{model_name}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3d(in_channels=4, n_classes=3,
               n_channels=32).to(device)
model.load_state_dict(torch.load(f'{save_path}/best-{model_name}.pth'))
torch.save(model, f'{save_path}/{model_name}.pth')
