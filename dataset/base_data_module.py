import copy
import math
import importlib
from pathlib import Path
import lightning as L
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from functools import partial
from typing import List, Optional, Dict, Union, Type, Tuple, Callable, Any
from .. import dynamic_import
from .pipeline import compose
import functools
import time
import os


def collate_from_dict(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    x, gt, info = [], [], []
    for d in batch:
        x.append(torch.from_numpy(d.pop('img')))
        gt.append(torch.from_numpy(d.pop('gt_seg_map')))
        # info.append(d)
    x = torch.stack(x).permute(0, 3, 1, 2).float()
    gt = torch.stack(gt).long()
    return x, gt, info


def collate_from_dict_semi(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    x, gt, info = [], [], []
    for ds in batch:
        for d in ds:
            x.append(torch.from_numpy(d.pop('img')))
            gt.append(torch.from_numpy(d.pop('gt_seg_map')))
            # info.append(d)
    x = torch.stack(x).permute(0, 3, 1, 2).float()
    gt = torch.stack(gt).long()
    return x, gt, info


def collate_from_dict_scribble(batch, *,
                               collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    x, gt, info = [], [], []
    for d in batch:
        x.append(torch.from_numpy(d.pop('img')))
        gt.append(torch.from_numpy(d.pop('gt_seg_map_scribble')))
        # info.append(d)

    x = torch.stack(x).permute(0, 3, 1, 2).float()
    gt = torch.stack(gt).long()

    return x, gt, info


class DistDataset(Dataset):
    def __init__(self, dataset: Dataset, world_size: int, sample: Any):
        self.dataset: Dataset = dataset
        self.world_size = world_size
        self.sample = sample
        self.length = len(dataset)
        self.dist_length = math.ceil(self.length / world_size) * world_size

    def __len__(self):
        return self.dist_length

    def __getitem__(self, idx):
        if idx < self.length:
            return self.dataset[idx]
        else:
            return self.sample


class LitDataModule(L.LightningDataModule):
    def __init__(self, name, batch_size=2, num_workers=8, seed=123456789,
                 multiprocessing_context=None, pin_memory=False, prefetch_factor=None, persistent_workers=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        if self.name in ['ade20k', 'ade']:
            from .ade20k import get_config
        elif self.name in ['cityscapes', 'city']:
            from .cityscapes import get_config
        elif self.name in ['voc_aug', 'voc']:
            from .voc2012_aug import get_config
        elif self.name in ['voc_aug_scribblesup', 'voc_aug_ss', 'voc_ss']:
            from .voc2012_aug_scribblesup import get_config
        elif self.name in ['kitti360', 'kit']:
            from .kitti360 import get_config
        else:
            raise ValueError(f'{self.name} is not supported, supported ade20k cityscapes city voc_aug')
        if 'crop_size' in kwargs:
            crop_size = kwargs['crop_size']
            if isinstance(crop_size, int):
                kwargs['crop_size'] = (crop_size, crop_size)
        self.cfg = copy.deepcopy(get_config(batch_size=batch_size, num_workers=num_workers, **kwargs))
        self.num_classes = self.cfg[0]
        self.train_dataloader_cfg = self.cfg[3]
        self.val_dataloader_cfg = self.cfg[6]

        self.train_dataloader_cfg['multiprocessing_context'] = multiprocessing_context
        self.train_dataloader_cfg['pin_memory'] = pin_memory
        self.train_dataloader_cfg['prefetch_factor'] = prefetch_factor
        self.train_dataloader_cfg['persistent_workers'] = persistent_workers
        self.val_dataloader_cfg['persistent_workers'] = persistent_workers
        if 'raw' in kwargs:
            self.train_dataloader_cfg['shuffle'] = False

        self.METAINFO = dict(
            classes=('background', 'aeroplane'),
            palette=[[0, 0, 0], [128, 0, 0], ])

    def train_dataloader(self):
        self.train_dataloader_obj = self.build_dataloader(self.train_dataloader_cfg)
        return self.train_dataloader_obj

    def val_dataloader(self):
        if 'WORLD_SIZE' in os.environ:
            self.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
            self.val_dataloader_cfg['WORLD_SIZE'] = self.WORLD_SIZE
        if self.val_dataloader_cfg is not None:
            self.val_dataloader_cfg['num_workers'] = self.num_workers // 2
            self.val_dataloader_obj = self.build_dataloader(self.val_dataloader_cfg)
            return self.val_dataloader_obj

    def train_dataset(self):
        return self.build_dataset(self.train_dataloader_cfg['dataset'])

    def val_dataset(self):
        return self.build_dataset(self.val_dataloader_cfg['dataset'])

    def build_dataset(self, cfg: dict, WORLD_SIZE: int = None):
        if isinstance(cfg, list):
            dataset = ConcatDataset([dynamic_import(**d) for d in cfg])
            self.METAINFO = dataset.datasets[0].METAINFO
        else:
            dataset = dynamic_import(**cfg)
            self.METAINFO = dataset.METAINFO
        print(f'{len(dataset)=}')
        if WORLD_SIZE is not None:
            dataset = DistDataset(dataset, world_size=WORLD_SIZE, sample=dict(
                img=np.zeros((320, 320, 3)),
                gt_seg_map=np.full((320, 320), fill_value=255, dtype=np.uint8),
                gt_seg_map_scribble=np.full((320, 320), fill_value=255, dtype=np.uint8),
            ))
            print(f'DistDataset: {len(dataset)=}')
        return dataset

    def build_dataloader(self, cfg: dict):
        assert isinstance(cfg, dict), "cfg must be a dict"
        cfg = copy.deepcopy(cfg)
        dataset = cfg['dataset']
        WORLD_SIZE = cfg.pop('WORLD_SIZE', None)
        cfg['dataset'] = self.build_dataset(dataset, WORLD_SIZE)
        if 'collate_fn' in cfg:
            cfg['collate_fn'] = dynamic_import(**cfg['collate_fn'], object=True)
        obj = DataLoader(**cfg)
        return obj
