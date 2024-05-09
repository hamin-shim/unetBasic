import pandas as pd
from utils import get_dataloader, dice_coef_metric_per_classes, compute_scores_per_classes
from dataset import BratsDataset
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Arrange GPU devices starting from 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"  # Set the GPUs 2 and 3 to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dice_score_excel(model_info):
    model_name = model_info['model_name']
    test_dataloader = get_dataloader(dataset=BratsDataset, phase="test",
                                     resize_info=model_info['resize_info'], img_width=model_info['img_size'], data_type=model_info['used_channel'], batch_size=2)
    model = torch.load(os.path.join('models', model_name,
                       f"{model_name}.pth")).to(device)
    model.eval()
    total = {'ID': [], 'WT': [], 'TC': [], 'ET': []}
    with torch.no_grad():
        for itr, data_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'Test Set'):
            # if(itr==2): break
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
            total['ET'] += res['ET']
            total['TC'] += res['TC']
            del images, targets
    df = pd.DataFrame(total)
    df.set_index('ID')
    df.to_excel(os.path.join('models', model_name,
                'logs', 'dice_score_inference.xlsx'))


def make_img(model_info, threshold=0.33):
    model_name = model_info['model_name']
    test_dataloader = get_dataloader(dataset=BratsDataset, phase="test",
                                     resize_info=model_info['resize_info'], img_width=model_info['img_size'], data_type=model_info['used_channel'], batch_size=4)
    model = torch.load(os.path.join('models', model_name,
                       f"{model_name}.pth")).to(device)
    model.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model, model_name, test_dataloader, ['WT', 'TC', 'ET'], threshold
    )
    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    # CONCAT BOTH THE COLUMNS ALONG AXIS 1 & SORT THE TWO
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard',
                                          'TC dice', 'TC jaccard',
                                          'ET dice', 'ET jaccard']]
    val_metics_df.to_excel(f'models/{model_name}/total_scores.xlsx')
    colors = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#e76f51']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=val_metics_df.mean().index,
                y=val_metics_df.mean(), palette=palette, ax=ax)
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title(f"{model_name}", fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig(f"figs/{model_name}_{threshold}.png", format="png",
                pad_inches=0.2, transparent=False, bbox_inches='tight')


with open('models/model_subscriptions.json', 'r') as f:
    models_info = json.load(f)
threshold = 0.33
# print(models_info)
make_img(models_info[-2])
# for model_info in tqdm(models_info, desc='total model'):
#     model_name = model_info["model_name"]
#     try:
#         if (os.path.exists(f"figs/{model_name}_{threshold}.png")):
#             print(f"{model_name} Already exist, pass")
#         else:
#             make_img(model_info, threshold)
#             print(f"{model_name} Success")
#     except:
#         print(f"{model_name} went wrong")

# /home/jiwoo/문서/GitHub/MUvit/figs
