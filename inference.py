import os
import torch
import json
import torch.nn as nn
import os
from utils import get_dataloader, load_test_dataset
from dataset import BratsDataset
import numpy as np
from tqdm import tqdm
from eval_utils import compute_dice
from skimage.transform import resize
import pandas as pd

# Arrange GPU devices starting from 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set the GPUs 2 and 3 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

with open('models/model_subscriptions.json', 'r') as f:
    models_info = json.load(f)
models_list = os.listdir('models')
print({i: c for i, c in enumerate(models_list)})
model_name = models_list[int(input("Choose model to inference: "))]
model_info = next(
    (model for model in models_info if model['model_name'] == model_name), None)
print(model_info)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = torch.load(os.path.join('models', model_name, f"{model_name}.pth"))
_model = _model.to(device)
model = nn.DataParallel(_model).to(device)
model.eval()

test_dataloader = get_dataloader(dataset=BratsDataset, phase="test",
                                 resize_info=model_info['resize_info'], img_width=model_info['img_size'], data_type=model_info['used_channel'], batch_size=4)
# Check dataloader
# test_batch = next(iter(test_dataloader))
# batch_id, images, targets = test_batch['Id'], test_batch['image'], test_batch['mask']
# images = images.to(device)
# targets = targets.to(device)
# print('batch id', batch_id)
# print('loaded image, target shape', images.shape, targets.shape)

dice_scores = []
_df = []
with torch.no_grad():
    for itr, data_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'Test Set'):
        batch_id, images, targets = data_batch['Id'], data_batch['image'], data_batch['mask']
        images = images.to(device)
        logits = model(images)
        pred = torch.sigmoid(logits).detach().cpu().numpy()
        threshold = 0.33
        pred = (pred >= threshold).astype(int)
        pred = np.array([resize(_pred, (3, 155, 240, 240), preserve_range=True)
                         for _pred in pred])
        if (pred.shape == targets.shape):
            dice_score = compute_dice(targets, pred)
            _df.append({'batch_id': batch_id, 'score': dice_score})
            dice_scores.append(dice_score)
        else:
            print(pred.shape, targets.shape)
        del images, targets, logits, pred
        torch.cuda.empty_cache()
dice_scores = np.array(dice_scores)
print("Total Dice score:", dice_scores.mean())
df = pd.DataFrame(_df)
df.to_excel(os.path.join('models', model_name,
            'logs', 'dice_score_inference.xlsx'))
