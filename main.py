from model import UNet3d
import torch
from train import Trainer
from dataset import BratsDataset
from utils import BCEDiceLoss
import json
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
model_name = input('Model name to save:')
epoch = int(input('Epoch: '))
batch_size = int(input('Batch size: '))
resize_info = [int(_) for _ in input(
    "Enter resizing start, end, interval: ").split(',')]
resize_data = [[cut_idx, real_idx]
               for cut_idx, real_idx in enumerate(range(*resize_info))]
print(f"Data will be resized to ({len(resize_data)},120,120)")
with open(f'saved_model/slicing_map_{model_name}.json', 'w') as f:
    json.dump(resize_data, f)
print(f"Slicing map is saved at saved_model/slicing_map_{model_name}.json")
trainer = Trainer(net=model,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=6,
                  batch_size=batch_size,
                  num_epochs=epoch,
                  path_to_log='logs', model_name=model_name, resize_info=resize_info)
start_time = time()
trainer.run()
end_time = time()
model_scription = {
    'model_name': model_name,
    'epoch': epoch,
    'batch_size': batch_size,
    'resize_info': resize_info,
    'resized_depth': len(resize_data),
    'best_model_path': f'saved_model/best-{model_name}.pth',
    'latest_model_path': f'saved_model/latest-{model_name}.pth',
    'csv_log_path': f'logs/train_log({model_name}).csv',
    'plot_path': f'logs/plot-{model_name}.jpg',
    'resize map path': f'saved_model/slicing_map_{model_name}.json',
    'run time(s)': end_time-start_time
}
with open('saved_model/model_subscriptions.json', 'r') as f:
    data = json.load(f)
data.append(model_scription)
with open('saved_model/model_subscriptions.json', 'w') as f:
    json.dump(data, f)
