from model import UNet3d
import torch
from train import Trainer
from dataset import BratsDataset
from utils import BCEDiceLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3d(in_channels = 3, n_classes = 3, n_channels = 32).to(device)
model_name = input('Model name to save:')
epoch = int(input('Epoch: '))
trainer = Trainer(net=model,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=4,
                  num_epochs=epoch,
                  path_to_log= 'logs', model_name = model_name)
trainer.run()