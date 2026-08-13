# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import numpy as np
import pdb 

from .transforms import RandomErasing

class ImageAligned(object):
    def __init__(self, size, mean=[0,0,0]):
        self.size = size
        self.mean = [int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)]

    def __call__(self, img):
        # resize image
        wimg, himg = T.functional.get_image_size(img)
        img = T.functional.resize(img, self.size-1, max_size=self.size)
        wimg, himg = T.functional.get_image_size(img)
        if wimg < self.size:
            padsize = (self.size - wimg) / 2
            img = T.functional.pad(img, [int(padsize), 0], fill=self.mean)
        else:
            padsize = (self.size - himg) / 2
            img = T.functional.pad(img, [0, int(padsize)], fill=self.mean)


        #print(np.asarray(img))

        return img


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        if cfg.INPUT.ALIGNED == 'yes':
            transform = T.Compose([
                ImageAligned(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.PIXEL_MEAN),
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        elif cfg.INPUT.AUTOAUGMENT == 'yes':
            transform = T.Compose([
                T.AutoAugment(),
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        elif cfg.INPUT.RANDAUGMENT == 'yes':
            transform = T.Compose([
                T.RandAugment(),
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
    else:
        if cfg.INPUT.ALIGNED == 'yes':
            transform = T.Compose([
                ImageAligned(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.PIXEL_MEAN),
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])


    return transform
