import copy
import importlib
import random
from pathlib import Path
import lightning as L
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from functools import partial
from typing import List, Optional, Dict, Union, Type, Tuple, Callable
from .. import dynamic_import
from .pipeline import compose


class BaseDataset(Dataset):
    METAINFO = dict(
        classes=('background', 'classes',),
        palette=[[0, 0, 0], [128, 0, 0], ]
    )

    def __init__(self, data_root, img_path_prefix: str = '', seg_map_path_prefix: str = '',
                 img_suffix='.jpg', seg_map_suffix='.png',
                 ann_file: str = None, pipeline: list = None):
        """
        初始化数据集。
        :param data_root: 数据集根目录。
        :param img_path_prefix: 图片文件相对路径前缀。
        :param seg_map_path_prefix: 分割标签相对路径前缀。
        :param ann_file: 存储图片 ID 的文件路径，例如 train.txt 或 val.txt。
        :param pipeline: 数据处理的操作（在实现中这里是配置，不实际用transform）。
        """
        self.data_root = data_root
        self.img_path_prefix = str(Path(img_path_prefix))
        self.seg_map_path_prefix = str(Path(seg_map_path_prefix))
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ann_file = ann_file
        self.pipeline = pipeline if pipeline is not None else []
        self.data_list: List[dict] = []

        self.init()
        self.sorted_by_info()  # Fixed order
        self.pipeline = compose(self.pipeline)

    def sorted_by_info(self):
        self.data_list.sort(key=lambda x: (x['img_path'], x['seg_map_path']))

    def init(self):
        if self.ann_file is None:
            _suffix_len = len(self.img_suffix)
            img_fps = sorted((Path(self.data_root) / self.img_path_prefix).glob(f'**/*{self.img_suffix}'))
            for img_fp in img_fps:
                seg_map_fp = (str(img_fp).replace(self.img_path_prefix, self.seg_map_path_prefix)
                              .replace(self.img_suffix, self.seg_map_suffix))
                self.data_list.append({'img_path': str(img_fp), 'seg_map_path': str(seg_map_fp),
                                       'seg_fields': [], 'reduce_zero_label': None})

        else:
            # 读取文件列表
            ann_file_path = Path(self.data_root) / self.ann_file
            with open(ann_file_path, 'r') as f:
                image_ids = [line.strip() for line in f]
            for image_id in image_ids:
                img_path = (Path(self.data_root) / self.img_path_prefix / image_id).with_suffix('.jpg')
                seg_map_fp = (Path(self.data_root) / self.seg_map_path_prefix / image_id).with_suffix('.png')
                self.data_list.append({'img_path': str(img_path), 'seg_map_path': str(seg_map_fp),
                                       'seg_fields': [], 'reduce_zero_label': None})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_list[idx])
        results['sample_idx'] = idx
        for p in self.pipeline:
            results = p(results)
        return results
    # def __getitem__(self, idx):
    #     results = copy.deepcopy(self.data_list[idx])
    #     results['sample_idx'] = idx
    #
    #     rt_list = []
    #     ts = time.time()
    #
    #     for p in self.pipeline:
    #         results = p(results)
    #         te = time.time()
    #         rt_list.append(round(te - ts, 6))
    #         ts = te
    #     m = np.array(rt_list).argmax()
    #     rt_list = [sum(rt_list)] + rt_list
    #     print(rt_list, m)
    #     return results


class SemiDataset(Dataset):
    METAINFO: dict

    def __init__(self, labeledtxt, num_unlabeled=1, dataset_cfg=None):

        self.dataset_cfg = dataset_cfg
        self.dataset: BaseDataset = self.build_dataset(dataset_cfg)
        self.labeledtxt = labeledtxt

        self.num_labeled = 1
        self.num_unlabeled = 1

        with open(str(Path(__file__).parent / 'splits' / labeledtxt), 'r') as f:
            d = f.readlines()
        labeled_str = [i.split(' ')[0] for i in d]

        data_list = self.dataset.data_list
        self.labeled_idx = []
        idx_data_list = list(range(len(data_list)))

        for lstr in labeled_str:
            for i in idx_data_list:
                infos = data_list[i]
                img_path = str(infos['img_path'])
                if lstr in img_path:
                    self.labeled_idx.append(i)
                    idx_data_list.remove(i)
                    break
        assert len(self.labeled_idx) == len(labeled_str)
        self.unlabeled_idx = list(set(range(len(data_list))) - set(self.labeled_idx))

    def __len__(self):
        return len(self.labeled_idx) // self.num_labeled

    def __getitem__(self, item):
        result_list = []
        for i in range(self.num_labeled):
            idx = random.choice(self.labeled_idx)
            result_list.append(self.dataset[idx])
        for i in range(self.num_unlabeled):
            idx = random.choice(self.unlabeled_idx)
            rsl=self.dataset[idx]
            rsl['gt_seg_map']=np.full_like(rsl['gt_seg_map'],fill_value=254)
            result_list.append(rsl)
        return result_list

    def build_dataset(self, cfg: dict):
        if isinstance(cfg, list):
            dataset = ConcatDataset([dynamic_import(**d) for d in cfg])
            self.METAINFO = dataset.datasets[0].METAINFO
        else:
            dataset = dynamic_import(**cfg)
            self.METAINFO = dataset.METAINFO
        return dataset
