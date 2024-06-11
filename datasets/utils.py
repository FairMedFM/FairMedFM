import numpy as np
import torch
import torchvision.transforms as transforms
import datasets
import pandas as pd
import random
import torchio as tio

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from torch.utils.data import WeightedRandomSampler


class ToTensor(object):

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor (C, H, W, T).
        Returns:
            Tensor: Converted image. (C, T, H, W)
        """

        img = (img - img.min()) / (img.max() - img.min())
        img = img.transpose(0, 3, 1, 2)
        img = torch.FloatTensor(img)
        return img

    def randomize_parameters(self):
        pass


def get_transform(args, split, augment=False):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.task == "cls":
        if split == "train":
            if augment:
                transform = transforms.Compose(
                    [
                        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation((-15, 15)),
                        transforms.RandomCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    elif args.task == "seg":
        # TODO
        pass
    else:
        raise NotImplementedError

    return transform


def get_dataset(args, split):
    data_setting = args.data_setting

    transform = get_transform(args, split, augment=args.augment)

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # image_path = data_setting["image_feature_path"]
    meta = pd.read_csv(data_setting[f"{split}_meta_path"])

    dataset_name = getattr(datasets, args["dataset_name"])
    image_path = None

    if args.task == "cls":
        data = dataset_name(meta, args.sensitive_name, transform, path_to_images=image_path)
    elif args.task == "seg":
        # TODO
        pass
    else:
        raise NotImplementedError

    print("loaded dataset ", args.dataset_name)

    if split == "train":
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(args.method != "resampling"),
            num_workers=6,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=6,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    return data, data_loader, meta
