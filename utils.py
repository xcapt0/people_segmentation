import cv2
import numpy as np
import albumentations as A
from sklearn.externals._pilutil import bytescale


def load_augs(aug_type):
    if aug_type == 'train':
        return A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(p=0.3),
            A.RandomGamma(p=0.5),
            A.Resize(224, 224)
        ])
    elif aug_type == 'test':
        return A.Compose([
            A.Resize(224, 224)
        ])


def min_max_scaler(img):
    return (img - np.min(img)) / np.ptp(img)


def inverse_scaler(img):
    scaled = bytescale(img, low=0, high=255)
    scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    return scaled
