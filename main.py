from model import UNet3d
import torch
from train import Trainer
from dataset import BratsDataset
from utils import BCEDiceLoss
import json
from time import time
import os
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPUs 2 and 3 to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load model by state dict
data_type = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2f.nii.gz']
# data_type = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz']
in_channel = len(data_type)
model_name = input('Model name to save:')
n_channel = int(input("n channel?: "))
model = UNet3d(in_channels=in_channel, n_classes=3,
               n_channels=n_channel).to(device)
# model = nn.DataParallel(_model).to(device)

save_path = f'models/{model_name}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)

# Config model (epoch, batch, image resizing info)
epoch = int(input('Epoch: '))
batch_size = int(input('Batch size: '))
img_width = int(input("Resized img width and height?: "))
resize_info = [int(_) for _ in input(
    "Enter resizing start, end, interval: ").split(',')]
resize_data = [[cut_idx, real_idx]
               for cut_idx, real_idx in enumerate(range(*resize_info))]
print(f"Data will be resized to ({len(resize_data)},{img_width},{img_width})")

# Save slicing map info
with open(f'{save_path}/mapping_{model_name}.json', 'w') as f:
    json.dump(resize_data, f)

trainer = Trainer(net=model, data_type=data_type,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=batch_size,
                  num_epochs=epoch,
                  path_to_log=os.path.join(save_path, 'logs'), model_name=model_name, resize_info=resize_info, img_width=img_width)

start_time = time()
trainer.run()
end_time = time()

# After train is done, save model(with best model)
model.load_state_dict(torch.load(f'{save_path}/best-{model_name}.pth'))
torch.save(model, f'{save_path}/{model_name}.pth')

# Save model description
model_scription = {
    'model_name': model_name,
    'in_channel': in_channel,
    'n_channel': n_channel,
    'img_size': img_width,
    "used_channel": data_type,
    'val score(dice/jaccard)': [int(trainer.dice_scores['val'][-1]*100), int(trainer.jaccard_scores['val'][-1]*100)],
    'batch/total epoch/best epoch': [batch_size, epoch, trainer.best_epoch],
    'run time(m)': (end_time-start_time)//60,
}
with open('models/model_subscriptions.json', 'r') as f:
    data = json.load(f)
data.append(model_scription)
with open('models/model_subscriptions.json', 'w') as f:
    json.dump(data, f)
