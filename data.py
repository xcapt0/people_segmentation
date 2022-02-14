import cv2
from glob import glob
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from utils import min_max_scaler


class SegmentationTrain(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.img_paths = imgs
        self.mask_paths = masks
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.load(self.img_paths[idx])
        mask = self.load(self.mask_paths[idx], image=False)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = min_max_scaler(img).transpose(2, 0, 1)
        mask = min_max_scaler(mask)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

    @staticmethod
    def load(path, image=True):
        img = cv2.imread(path)

        if image:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_paths():
    images = glob(f'AHP/train/JPEGImages/*')
    masks = glob(f'AHP/train/Annotations/*')
    return images, masks


def split(images, masks, test_size=0.2):
    images.sort()
    masks.sort()
    return train_test_split(images, masks, test_size=test_size, random_state=42)
