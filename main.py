from model import UNet3d
import torch
from train import Trainer
from dataset import BratsDataset
from utils import BCEDiceLoss
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3d(in_channels = 3, n_classes = 3, n_channels = 32).to(device)
model_name = input('Model name to save:')
epoch = int(input('Epoch: '))
batch_size = int(input('Batch size: '))
trainer = Trainer(net=model,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=batch_size,
                  num_epochs=epoch,
                  path_to_log= 'logs', model_name = model_name)
trainer.run()

model_scription = {
    'model_name':model_name, 
    'epoch':epoch, 
    'batch_size':batch_size,
    'best_model_path':f'saved_model/best-{model_name}.pth',
    'latest_model_path':f'saved_model/latest-{model_name}.pth',
    'csv_log_path': f'logs/train_log({model_name}).csv',
    'plot_path': f'logs/plot-{model_name}.jpg'
}
with open('saved_model/model_subscriptions.json','r') as f:
    data = json.load(f)
data.append(model_scription)
with open('saved_model/model_subscriptions.json','w') as f:
    json.dump(data, f)