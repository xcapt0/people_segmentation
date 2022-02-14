import torch
from argparse import ArgumentParser

from data import load_paths, split


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        from train import Trainer

        images, masks = load_paths()
        train_data = split(images, masks)

        trainer = Trainer(*train_data, checkpoint=args.checkpoint, device=device)
        trainer.train()
    elif args.segment:
        from transform import Segmentation

        segmentation = Segmentation(args, device=device)
        segmentation.segment()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true',
                        help='Training mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint of the model')
    parser.add_argument('--segment', type=str, default=None,
                        help='Segmentation mode. Path of the image or the directory')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save segmented image')
    user_args = parser.parse_args()

    main(user_args)
