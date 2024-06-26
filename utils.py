import torch
from torch.utils.data import DataLoader
# from albumentations import Compose
import os
import numpy as np
import nibabel as nib
import torch.nn as nn
from tqdm import tqdm
# import albumentations as A
from volumentations import *


def get_augmentations(phase):
    list_transforms = []
    # if (phase == 'train'):
    #     list_transforms = [
    #         Flip(p=0.25),
    #         RandomRotate90((2, 3), p=0.25),
    #         GaussianNoise(var_limit=(0, 0.002), p=0.25),
    #         RandomGamma(gamma_limit=(80, 120), p=0.25),
    #         ElasticTransform((0, 0.25), interpolation=2, p=0.25),
    # HorizontalFlip(p=0.3),
    # VerticalFlip(p=0.3),
    # GaussianBlur(p=0.3),
    # ]
    # Does data augmentations & tranformation required for IMAGES & MASKS
    # they include cropping, padding, flipping , rotating
    list_trfms = Compose(list_transforms)
    # list_trfms = Compose(list_transforms, is_check_shapes=False)
    return list_trfms


def get_dataloader(
        dataset: torch.utils.data.Dataset,
        phase: str,
        img_depth: int, img_width: int, data_type: list,
        batch_size: int = 4,
        num_workers: int = 4):
    # ids = os.listdir(os.path.join("brats_data", phase))
    ids = ['BraTS-GLI-00715-001', 'BraTS-GLI-01085-000', 'BraTS-GLI-01161-000']
    ds = dataset(data_path='brats_data', data_type=data_type, phase=phase, ids=ids, img_width=img_width,
                 is_resize=True, img_depth=img_depth)
    """
    DataLoader iteratively goes through every id in the df & gets all the individual tuples for individual ids & appends all of them 
    like this : 
    { id : ['BraTS-GLI-00000-000'] ,
      image : [] , 
      tensor : [] , 
    } 
    { id : ['BraTS-GLI-00000-000'] ,
      image : [] , 
      tensor : [] , 
    }
    """
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    '''factory for storing and updating iou and dice scores.'''

    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        calculates using the above declare functions 
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)

        # appending to the respective lists
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou  # type: ignore


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert (probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        # print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # logits are the images
        # target are the masks
        assert (logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        # binary cross entropy loss & dice loss
        return bce_loss + dice_loss

# helper functions for testing.


def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                 truth: np.ndarray,
                                 treshold: float = 0.5,
                                 eps: float = 1e-9,
                                 classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class i.e. 'WT', 'TC', 'ET'
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,  # output of the model in an array format
                                    truth: np.ndarray,  # masks
                                    treshold: float = 0.5,  # threshold to whether segment / not
                                    eps: float = 1e-9,  # smooth
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Jaccard index for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    # storing all the jaccard coefficients in a list

    num = probabilities.shape[0]

    num_classes = probabilities.shape[1]

    # segmenting if prob > threshold .i.e. setting to float32
    predictions = (probabilities >= treshold).astype(np.float32)

    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def preprocess_mask_labels(mask):
    # whole tumour
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1
    # include all tumours

    # NCR / NET - LABEL 1
    mask_TC = mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1
    # exclude 2 / 4 labelled tumour

    # ET - LABEL 3
    mask_ET = mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1
    # exclude 2 / 1 labelled tumour

    # mask = np.stack([mask_WT, mask_TC, mask_ET, mask_ED])
    mask = np.stack([mask_WT, mask_TC, mask_ET])
    return mask


def load_test_dataset(id_):
    data = nib.load(os.path.join('brats_data', 'test', id_, id_+'-seg.nii.gz'))
    data = np.asarray(data.dataobj)  # (240,240,155)
    data = data.transpose(2, 0, 1)
    return preprocess_mask_labels(data)


def compute_scores_per_classes(model, model_name,          # nodel which is UNeT3D
                               # tuple consisting of ( id , image tensor , mask tensor )
                               dataloader,
                               classes, threshold=0.33):       # classes : WT , TC , ET
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=model_name):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probabilities = torch.sigmoid(logits)
            prediction = (probabilities >= threshold).float()
            prediction = prediction.detach().cpu().numpy()

            targets = targets.detach().cpu().numpy()

            # Now finding the overlap between the raw prediction i.e. logit & the mask i.e. target & finding the dice & iou scores
            dice_scores = dice_coef_metric_per_classes(prediction, targets)
            iou_scores = jaccard_coef_metric_per_classes(prediction, targets)

            # storing both dice & iou scores in the list declared
            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes
