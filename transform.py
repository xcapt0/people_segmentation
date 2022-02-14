import os
import cv2
import numpy as np
import torch
from glob import glob
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from model import UNet
from utils import min_max_scaler, inverse_scaler


class Segmentation:
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.model = self._load_model()
        self.images = self._load_images()

    def segment(self, plot=False, save_dir='./tmp'):
        if self.args.save_dir:
            save_dir = self.args.save_dir

        self.model.eval()

        for path, image in tqdm(self.images, desc='Image'):
            image = torch.unsqueeze(image, 0).to(self.device)

            with torch.no_grad():
                output = self.model(image)

            output = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(output, dim=1)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = np.squeeze(pred_mask)
            image = image.cpu().numpy()[0].transpose(1, 2, 0)

            if self._are_sizes_different(image, pred_mask):
                mask_h, mask_w = pred_mask.shape
                image = cv2.resize(image, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

            masked = self._apply_mask(image, pred_mask)

            if plot:
                self._plot(masked)

            self._save(masked, path, save_dir)

    def _load_model(self):
        path = self.args.checkpoint

        if os.path.isfile(path):
            model = UNet(in_channels=3, out_channels=2, n_blocks=4, start_filters=32,
                         activation='relu', normalization='batch', conv_mode='same',
                         dim=2).to(self.device)

            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise FileNotFoundError('Model checkpoint not found')

    def _load_images(self):
        files = self._get_files()

        if len(files) <= 0:
            raise FileNotFoundError('Images not found')

        for path in files:
            image = cv2.imread(path)
            image = self._transform(image)
            yield path, image

    def _get_files(self):
        path = self.args.segment
        _, extension = os.path.splitext(path)

        if extension:
            return glob(path)
        else:
            files = []
            for filetype in ['png', 'jpg', 'jpeg']:
                files.extend(glob(os.path.join(path, f'*.{filetype}')))
            return files

    @staticmethod
    def _transform(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = min_max_scaler(image).transpose(2, 0, 1)
        return torch.from_numpy(image).float()

    @staticmethod
    def _plot(image):
        plt.imshow(image)
        plt.show()

    @staticmethod
    def _save(image, path, save_dir):
        if os.path.isdir(save_dir):
            image = inverse_scaler(image)
            filename = os.path.basename(path)
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, image)
        else:
            os.makedirs(save_dir)

    @staticmethod
    def _are_sizes_different(image, mask):
        img_h, img_w, _ = image.shape
        mask_h, mask_w = mask.shape
        if img_w != mask_w or img_h != mask_h:
            return True
        return False

    @staticmethod
    def _apply_mask(image, mask):
        color = np.array([1, 0, 1], dtype='uint8')
        masked_img = np.where(mask[..., None], color, image)
        return cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
