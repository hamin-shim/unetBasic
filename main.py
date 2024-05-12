from model import UNet3d
import torch
from train import Trainer
from dataset import BratsDataset
from utils import BCEDiceLoss
import json
from time import time
import os
import torch.nn as nn
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"  # Set the GPUs 2 and 3 to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config model (epoch, batch, image resizing info)
epoch = 100
batch_size = 4
img_depth = 64
img_width = 128
n_channel = 32
data_type = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2f.nii.gz']
# data_type = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz']
in_channel = len(data_type)
model_feature = input('Model feature to name:')
model_name = f"{model_feature}_n{n_channel}_d{img_depth}_w{img_width}_b{batch_size}_e{epoch}"
_model = UNet3d(in_channels=in_channel, n_classes=3,
                n_channels=n_channel).to(device)
model = nn.DataParallel(_model).to(device)

save_path = f'models/{model_name}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)

trainer = Trainer(net=model, data_type=data_type,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=batch_size,
                  num_epochs=epoch,
                  path_to_log=os.path.join(save_path, 'logs'), model_name=model_name, img_depth=img_depth, img_width=img_width)

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
    'img_depth': img_depth,
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
