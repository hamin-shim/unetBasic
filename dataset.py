from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from utils import get_augmentations


class BratsDataset(Dataset):
    def __init__(self, data_path, data_type: list = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz'], phase: str = "train", ids: list = [], img_width=120, is_resize: bool = True, resize_info: list = [], aug=False):
        self.data_path = data_path
        self.phase = phase
        self.ids = ids
        if (aug):
            self.augmentations = get_augmentations('aug')
        else:
            self.augmentations = get_augmentations(phase)
        self.data_types = data_type
        # self.data_types = ['-t1n.nii.gz',
        #                    '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz']
        self.is_resize = is_resize
        self.resize_info = resize_info
        self.img_width = img_width

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # at a specified index ( idx ) select the value under 'Brats20ID' & asssign it to id_
        id_ = self.ids[idx]
        # print(id_)

        # load all modalities
        images = []

        for data_type in self.data_types:
            # here data_type is appended to the root path, as it only contains the name without the datatype such as .nii etc
            img_path = os.path.join(
                self.data_path, self.phase, id_, id_+data_type)
            img = self.load_img(img_path)  # img.shape = (240,240,155)

            if self.is_resize:
                img = img.transpose(2, 0, 1)  # img.shape = (155,240,240)
                img = self.resize(img)  # img.shape = (78,120,120)

            img = self.normalize(img)
            images.append(img)

        # stacking all the t1 , t1ce , t2 flair files of a single ID in a stack
        img = np.stack(images)

        if self.phase != "test":
            mask_path = os.path.join(
                self.data_path, self.phase, id_, id_+'-seg.nii.gz')
            mask = self.load_img(mask_path)  # (240,240,155), data:0~3
            mask = mask.astype(np.uint8)

            if self.is_resize:
                # (155,240,240) -> (??,240,240) -> (3,??,240,240) -> (3,??,120,120)
                mask = mask.transpose(2, 0, 1)  # mask.shape = (155,240,240)
                mask = self.preprocess_mask_labels(mask)
                # mask = self.resize_mask(mask)
            augmented = self.augmentations(image=img.astype(np.float32),
                                           mask=mask.astype(np.float32))
            # Several augmentations / transformations like flipping, rotating, padding will be applied to both the images
            img = augmented['image']
            mask = augmented['mask']

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        else:
            mask_path = os.path.join(
                self.data_path, self.phase, id_, id_+'-seg.nii.gz')
            mask = self.load_img(mask_path)  # (240,240,155), data:0~3
            mask = mask.astype(np.uint8)
            mask = mask.transpose(2, 0, 1)  # mask.shape = (155,240,240)
            mask = self.preprocess_mask_labels(mask)
            augmented = self.augmentations(image=img.astype(np.float32),
                                           mask=mask.astype(np.float32))
            # Several augmentations / transformations like flipping, rotating, padding will be applied to both the images
            img = augmented['image']
            mask = augmented['mask']

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        # normalization = (each element - min element) / ( max - min )
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        start_num, end_num, interval = self.resize_info
        # (155,240,240) -> (??,240,240)
        data = data[np.arange(start_num, end_num, interval)]
        # (??,240,240) -> (??, 120, 120)
        data = resize(
            data, (data.shape[0], self.img_width, self.img_width), preserve_range=True)
        return data

    def resize_mask(self, data: np.ndarray):
        # Flow : (155,240,240) -> (??,240,240) -> (3,??,240,240) -> (3,??,120,120)
        start_num, end_num, interval = self.resize_info
        # (155,240,240) -> (??,240,240)
        _data = data[np.arange(start_num, end_num, interval)]
        # (??,240,240) -> (3,??,240,240)
        data = self.preprocess_mask_labels(_data)
        # (3,??,240,240) -> (3,??, 120, 120)
        data = resize(
            data, (data.shape[0], data.shape[1], self.img_width, self.img_width), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):

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

        # ET - LABEL 4
        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 3] = 1
        # exclude 2 / 1 labelled tumour

        # mask = np.stack([mask_WT, mask_TC, mask_ET, mask_ED])
        mask = np.stack([mask_WT, mask_TC, mask_ET])

        return mask
