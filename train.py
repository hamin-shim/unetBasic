import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from dataset import BratsDataset
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy_brats,
    save_predictions_as_imgs,
)
import numpy as np

LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "C:/Users/user/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train/"
VAL_IMG_DIR = "C:/Users/user/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/val/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = np.transpose(np.squeeze(data), (3, 0, 1, 2))
        targets = np.transpose(np.squeeze(targets), (2, 0, 1))
        data = data.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=4, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_ds = BratsDataset(
        image_dir=TRAIN_IMG_DIR
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    val_ds = BratsDataset(
        image_dir=VAL_IMG_DIR
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )
    print(1)
    check_accuracy_brats(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    print(2)
    for epoch in range(2):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy_brats(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()