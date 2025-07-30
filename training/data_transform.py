# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transform():
    transform_list = [
        ToImage(),
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)

class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, height=256, width=256, mode="train"):
        self.dataset = dataset
        if mode == "train":
            self.transform = A.Compose(transforms=[A.Resize(height, width),
                                                A.HorizontalFlip(p=0.5),
                                                ToTensorV2()])
        else:
            self.transform = A.Compose(transforms=[A.Resize(height, width),
                                                ToTensorV2()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mask, image = self.dataset[idx]
        if isinstance(mask, list):
            transformed = self.transform(image=image, masks=mask)
            mask = [m.unsqueeze(0).to(torch.float32) for m in transformed["masks"]]
        else:
            transformed = self.transform(image=image, mask=mask)
            mask = transformed["mask"].unsqueeze(0).to(torch.float32)
        return mask, transformed["image"].to(torch.float32)