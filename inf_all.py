import pandas as pd
from tqdm import tqdm
from dataset import BratsDataset
from utils import get_dataloader
import torch
import torch.nn as nn
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import dice_coef_metric_per_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with open('models/models.json', 'r') as f:
#     models_info = json.load(f)


def inference(model_name):
    model = torch.load(os.path.join('models', model_name,
                                    f"{model_name}.pth")).to(device)
    model.eval()
    test_dataloader = get_dataloader(dataset=BratsDataset, phase="test", img_depth=155, img_width=240, data_type=[
        "-t1n.nii.gz",
        "-t1c.nii.gz",
        # "-t2w.nii.gz",
        "-t2f.nii.gz"
    ], batch_size=1)

    total = {'ID': [], 'WT': [], 'TC': [], 'ET': []}

    with torch.no_grad():
        for itr, data_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'{model_name[:10]}'):
            batch_id, images, targets = data_batch['Id'], data_batch['image'], data_batch['mask']
            images = images.to(device)
            targets = targets.detach().numpy()
            logits = model(images)
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            threshold = 0.33
            pred = (pred >= threshold).astype(int)
            res = dice_coef_metric_per_classes(pred, targets)
            total['ID'] += batch_id
            total['WT'] += res['WT']
            total['TC'] += res['TC']
            total['ET'] += res['ET']
            del images, targets

    df = pd.DataFrame(total)
    df.set_index('ID')

    df.to_excel(os.path.join('final_res', 'th_33'+model_name +
                'dice_score_inference.xlsx'))

    # print(df['WT'].mean(), df['TC'].mean(), df['ET'].mean())
    total_mean = (df['WT'].mean() + df['TC'].mean() + df['ET'].mean())/3
    with open('final_res/total_log_th_33.txt', 'a') as f:
        f.write(
            f"[{df['WT'].mean():.3f} / {df['TC'].mean():.3f} / {df['ET'].mean():.3f}] ||  @{model_name}({total_mean:.3f})\n")
    return [df['WT'].mean(), df['TC'].mean(), df['ET'].mean()]


# for model_info in tqdm(models_info):
#     try:
#         res = inference(model_info)
#         print(
#             f"{model_info['model_name']} succesfully done, saved at final_res")
#     except:
#         print(f"{model_info['model_name']} has gone wrong, pass")
inference('ch3_32_interval_3_240')
