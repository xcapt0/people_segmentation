import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from model import UNet
from data import SegmentationTrain
from utils import load_augs


class Trainer:
    def __init__(self,
                 train_images,
                 train_masks,
                 valid_images,
                 valid_masks,
                 checkpoint=None,
                 device='cpu'):

        self.train_images = train_images
        self.train_masks = train_masks
        self.valid_images = valid_images
        self.valid_masks = valid_masks
        self.checkpoint_path = checkpoint
        self.device = device
        self._init_for_training()
        self.history = {'loss': [], 'val_loss': []}

    def train(self):
        if self.checkpoint_path:
            self._load_checkpoint()

        best_loss = 1e+5
        epochs = 40

        for epoch in range(epochs):
            # apply random dataset shuffle while loading
            self._load_dataset()

            # console logging while training
            dataset = tqdm(enumerate(self.train_loader), leave=False, total=len(self.train_loader))
            dataset.set_description(f'Epoch {epoch + 1}/{epochs}')

            self._train(dataset)
            self._validate()

            # save best loss model
            if round(self.val_loss, 4) < round(best_loss, 4):
                best_loss = self.val_loss
                self.save(f'best_model.pth')

            # print epoch training results
            self._print_epoch(dataset)
            self.plot_segmentation()

            self.model.train()
            self.scheduler.step()

    def plot_segmentation(self):
        img, mask = self.valid_dataset[random.randint(0, len(self.valid_dataset))]
        img = torch.unsqueeze(img, 0).to(self.device)
        mask = torch.unsqueeze(mask, 0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(img)

        output = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(output, dim=1)
        pred_mask = pred_mask.cpu().numpy()
        pred_mask = np.squeeze(pred_mask)

        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img.cpu().numpy()[0].transpose(1, 2, 0))
        axs[1].imshow(mask.cpu().numpy()[0])
        axs[2].imshow(pred_mask)
        plt.show()

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, filename)

    def remove_history(self):
        self.history = {'loss': [], 'val_loss': []}

    def _init_for_training(self):
        self.model = UNet(in_channels=3, out_channels=2, n_blocks=4, start_filters=32,
                          activation='relu', normalization='batch', conv_mode='same',
                          dim=2).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                           base_lr=1e-7,
                                                           max_lr=1e-2,
                                                           step_size_up=1,
                                                           mode="exp_range",
                                                           gamma=0.45,
                                                           cycle_momentum=False)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _load_dataset(self):
        self.train_dataset = SegmentationTrain(self.train_images, self.train_masks, transform=load_augs('train'))
        self.valid_dataset = SegmentationTrain(self.valid_images, self.valid_masks, transform=load_augs('test'))
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=64, shuffle=False, num_workers=2)

    def _train(self, dataset):
        for i, (inputs, masks) in dataset:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)

            output = self.model(inputs)
            loss = self.criterion(output, masks)

            self.history['loss'].append(loss.item())
            self.avg_loss = np.mean(self.history['loss'])

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            dataset.set_postfix_str(f'loss: {loss.item():.4f}')

    def _validate(self):
        self.model.eval()
        val_loss = []

        dataset = tqdm(self.valid_loader, desc=f'Evaluation', leave=False, total=len(self.valid_loader))
        for inputs, masks in dataset:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                output = self.model(inputs)

            loss = self.criterion(output, masks)
            val_loss.append(loss.item())

        self.val_loss = np.mean(val_loss)
        self.history['val_loss'].append(val_loss)
        self.avg_val_loss = np.mean(self.history['val_loss'])

    def _print_epoch(self, dataset):
        dataset.update()
        dataset.set_postfix_str(
            f'loss: {self.avg_loss:.4f} - val_loss: {self.avg_val_loss:.4f} - learning_rate: {self.scheduler.get_last_lr()[0]}')
        print(dataset)
