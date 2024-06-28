import os
import random

import albumentations as albu
import cv2
import einops
import ipdb
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from albumentations.augmentations import transforms as atransforms
from albumentations.core.composition import Compose
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms._transforms_video import NormalizeVideo

import datasets


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
                        transforms.Resize(
                            224, interpolation=transforms.InterpolationMode.BICUBIC),
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
                        transforms.Resize(
                            224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    elif args.task == "seg":
        transform = Compose([
            albu.Resize(args.img_size, args.img_size),
            atransforms.Normalize(),
            # transforms.ToTensor()
        ])
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
    # TODO: Add segmentation meta path
    meta = pd.read_csv(data_setting[f"{split}_meta_path"])

    if args.task == "cls":
        dataset_name = getattr(datasets, args.dataset)
        image_path = data_setting[f"image_{split}_path"]
        data = dataset_name(meta, args.sensitive_name,
                            transform, path_to_images=image_path)

    elif args.task == "seg":
        # TODO
        dataset = Dataset2D(
            basepath=data_setting["data_path"],
            pos_class=data_setting["pos_class"],
            transform=transform)
        data = DataEngine2D(
            dataset=dataset,
            img_size=(args.img_size, args.img_size)
        )
    else:
        raise NotImplementedError()

    print("loaded dataset ", args.dataset)

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
    elif split == "test":
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    return data, data_loader, meta


class Dataset2D(Dataset):
    """
        Dataset for 2D images
    """

    def __init__(self, basepath: str, pos_class=255, transform=None) -> None:
        self.basepath = basepath
        self.pos_class = pos_class
        self.meta = pd.read_csv(
            os.path.join(basepath, "metadata.csv"), index_col=0)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        sample = self.meta.iloc[idx].to_dict()

        image = cv2.imread(os.path.join(self.basepath, sample["imagePath"]))
        label = cv2.imread(os.path.join(self.basepath, sample["labelPath"]), cv2.IMREAD_GRAYSCALE)[
            ..., None]
        augmented = self.transform(image=image, mask=label)

        sample_dict = {
            "image": einops.rearrange(augmented["image"], "w h c -> c w h"),
            "label": einops.rearrange(augmented["mask"], "w h c -> c w h"),
            "sex": sample["sexBinary"][0],
            "filename": sample["labelPath"].split("/")[-1]
        }

        # ic(sample_dict["image"].shape, sample_dict["label"].shape)
        # Change range
        image = image.astype('float32')

        # print(np.unique(sample_dict["label"]))

        # Add mask spliting based on pos_class (using np.unique)
        sample_dict["label"] = np.uint8(
            sample_dict["label"] == self.pos_class)

        return sample_dict


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


class PointPromptGenerator(object):
    def __init__(self, size=None) -> None:
        self.size = size

    def get_prompt_point(self, gt_mask):
        # assert gt_mask.shape == (1024,1024) or gt_mask.shape == (512,512), f"[data_engine] {__file__} Got{gt_mask.shape}"
        if not (gt_mask.shape == (1024, 1024) or gt_mask.shape == (512, 512) or gt_mask.shape == (256, 256)):
            print(f"[Warning] [data_engine] {__file__} Got{gt_mask.shape}")
        assert gt_mask.sum() > 0
        self.size = gt_mask.shape
        self.xy = np.arange(0, self.size[0] * self.size[1])

        gt_mask = np.float32(gt_mask > 0)
        prob = rearrange(gt_mask, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        x, y = loc % self.size[1], loc // self.size[1]
        return x, y

    def get_prompt_center(self, gt_mask):
        # Find the indices of all non-zero elements in the mask
        coords = np.nonzero(gt_mask)

        # Compute the minimum and maximum values of the row and column indices
        x_min = np.min(coords[1])
        y_min = np.min(coords[0])
        x_max = np.max(coords[1])
        y_max = np.max(coords[0])

        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        # Return the coordinates of the bounding box
        return x, y

    def get_prompt_rands(self, gt_mask):
        # assert gt_mask.shape == (1024,1024) or gt_mask.shape == (512,512), f"[data_engine] {__file__} Got{gt_mask.shape}"
        if not (gt_mask.shape == (1024, 1024) or gt_mask.shape == (512, 512) or gt_mask.shape == (256, 256)):
            print(f"[Warning] [data_engine] {__file__} Got{gt_mask.shape}")
        assert gt_mask.sum() > 0
        self.size = gt_mask.shape
        self.xy = np.arange(0, self.size[0] * self.size[1])

        gt_mask = np.float32(gt_mask > 0)
        prob = rearrange(gt_mask, "h w -> (h w)")
        prob = prob / prob.sum()
        # ipdb.set_trace()
        # generate 5 random points
        loc = np.random.choice(a=self.xy, size=5, replace=True, p=prob)
        x, y = loc % self.size[1], loc // self.size[1]

        return x, y


class BoxPromptGenerator(object):
    def __init__(self, size) -> None:
        self.size = size

    @staticmethod
    def mask_to_bbox(mask):
        # Find the indices of all non-zero elements in the mask
        coords = np.nonzero(mask)

        # Compute the minimum and maximum values of the row and column indices
        x_min = np.min(coords[1])
        y_min = np.min(coords[0])
        x_max = np.max(coords[1])
        y_max = np.max(coords[0])

        # Return the coordinates of the bounding box
        return (x_min, y_min, x_max, y_max)
        # return (y_min, x_min, y_max, x_max)

    def add_random_noise_to_bbox(self, bbox):
        bbox = list(bbox)
        # Calculate the side lengths of the box in the x and y directions
        x_side_length = bbox[2] - bbox[0]
        y_side_length = bbox[3] - bbox[1]

        # Calculate the standard deviation of the noise
        std_dev = 0.01 * (x_side_length + y_side_length) / 2

        # Generate random noise for each coordinate
        x_noise = np.random.normal(scale=std_dev)
        y_noise = np.random.normal(scale=std_dev)

        # Add the random noise to each coordinate, but make sure it is not larger than 20 pixels
        bbox[0] += min(int(round(x_noise)), 20)
        bbox[1] += min(int(round(y_noise)), 20)
        bbox[2] += min(int(round(x_noise)), 20)
        bbox[3] += min(int(round(y_noise)), 20)

        # Make sure the modified coordinates do not exceed the maximum possible values
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], self.size[0])
        bbox[3] = min(bbox[3], self.size[1])

        # Return the modified bounding box
        return bbox

    def get_prompt_box(self, gt_mask):
        """ return (x_min, y_min, x_max, y_max) """
        assert gt_mask.shape == (1024, 1024) or gt_mask.shape == (512, 512) or gt_mask.shape == (
            256, 256), f"[data_engine] {__file__} Got{gt_mask.shape}"
        box = self.mask_to_bbox(gt_mask)
        # box_w_noise = self.add_random_noise_to_bbox(box)
        return box

    def enlarge(self, bbox, margin=0):
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        margin_x = int((x1 - x0)*0.05)
        margin_y = int((y1 - y0)*0.05)
        x0 = max(x0 - margin_x, 0)
        y0 = max(y0 - margin_x, 0)
        x1 = min(x1 - margin_y, self.size[0]-1)
        y1 = min(y1 - margin_y, self.size[1]-1)

        # print("[DEBUG] , enlarge size: ", margin_x, margin_y)
        # print("[DEBUG] from", bbox, "to", (x0,y0,x1,y1))
        return (x0, y0, x1, y1)


class DataEngine2D(Dataset):
    def __init__(self, dataset=None, img_size=None) -> None:
        # CACHE_DISK_DIR="/home1/quanquan/code/projects/medical-guangdong/cache/data2d_3/"
        super().__init__()
        self.point_prompt_generator = PointPromptGenerator(img_size)
        self.box_prompt_generator = BoxPromptGenerator(img_size)
        # self._get_dataset(dirpath=dirpath)
        self.dataset = dataset

    # def _get_dataset(self, dirpath):
    #     self.dataset = Dataset2D(dirpath=dirpath, is_train=True)

    def __len__(self):
        return len(self.dataset)

    def _get_true_index(self, idx):
        return idx

    def __getitem__(self, idx):
        return self.get_prompt(idx)

    def get_prompt_point(self, gt_mask):
        return self.point_prompt_generator.get_prompt_point(gt_mask)

    def get_prompt_box(self, gt_mask):
        return self.box_prompt_generator.get_prompt_box(gt_mask)

    def get_prompt_center(self, gt_mask):
        return self.point_prompt_generator.get_prompt_center(gt_mask)

    def get_prompt_rands(self, gt_mask):
        return self.point_prompt_generator.get_prompt_rands(gt_mask)

    # def _get_data_from_dataset(self, idx):

    def get_prompt(self, idx):
        idx = self._get_true_index(idx)
        data = self.dataset.__getitem__(idx)

        # img = data['img'] # (3,h,w) d=3
        gt_mask = data['label'][0]  # (1,h,w) d=3
        gt_mask = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask

        # check if the mask is empty
        if gt_mask.sum() == 0:
            return data

        # if np.random.rand() > 0.5:
        prompt_point = self.get_prompt_point(gt_mask)
        # else:
        prompt_box = self.get_prompt_box(gt_mask)
        prompt_center = self.get_prompt_center(gt_mask)
        prompt_rands = self.get_prompt_rands(gt_mask)

        '''
        Other keys of data: image, label, sex, filename
        '''

        data['prompt_point'] = np.array(prompt_point).astype(np.float32)
        data['prompt_box'] = np.array(prompt_box).astype(np.float32)
        data['prompt_center'] = np.array(prompt_center).astype(np.float32)
        data['prompt_rands'] = np.array(prompt_rands).astype(np.float32)

        # data['point_label'] = np.ones((1,)).astype(np.float32)

        return data
