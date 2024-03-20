import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class BratsDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        _img = []
        mask = None
        for nil_name in os.listdir(img_path):
            img_type = nil_name.split('-')[-1].split('.')[0]
            if img_type == 'seg':
                mask = nib.load(os.path.join(img_path, nil_name)).get_fdata()[:,:,50:100]
            else:
                _img.append(nib.load(os.path.join(img_path, nil_name)).get_fdata()[:,:,50:100])
        image = np.stack(_img)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(image)

        return image, mask