import os
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset

import transforms as T


class selfDataset(Dataset):
    def __init__(self, root: str, targetHeight: int, targetWidth: int, numClass: int, train: bool = True):
        """
        :param root: should contain .jpg files and corresponding .txt files
        :param targetHeight: desired height for model input
        :param targetWidth: desired width for model input
        :param numClass: number of classes in the given dataset
        """
        self.cache = {}

        imagePaths = glob(os.path.join(root, '*.jpg'))
        for path in imagePaths:
            name = path.split('/')[-1].split('\\')[-1].split('.jpg')[0]
            self.cache[path] = os.path.join(root, f'{name}.txt')

        self.paths = list(self.cache.keys())

        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.numClass = numClass

        if train:
            self.transforms = T.Compose([
                T.RandomOrder([
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomSizeCrop(numClass)
                ]),
                T.Resize((targetHeight, targetWidth)),
                T.ColorJitter(brightness=.2, contrast=.1,
                              saturation=.1, hue=0),
                T.Normalize()
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((targetHeight, targetWidth)),
                T.Normalize()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgPath = self.paths[idx]
        annPath = self.cache[imgPath]

        image = Image.open(imgPath).convert('RGB')
        annotations = self.loadAnnotations(annPath)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        image, targets = self.transforms(image, targets)

        return image, targets

    @staticmethod
    def loadAnnotations(path: str) -> np.ndarray:
        """
        :param path: annotation file path
                -> each line should be in the format of [class centerX centerY width height]

        :return: an array of objects of shape [centerX, centerY, width, height, class]
        """
        if not os.path.exists(path):
            return np.asarray([])

        ans = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                c, (x, y, w, h) = int(line[0]), list(map(float, line[1:]))
                ans.append([x, y, w, h, c])
        return np.asarray(ans)


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]
