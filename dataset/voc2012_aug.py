# from pathlib import Path
# import lightning as L
# from PIL import Image
# import numpy as np
# from typing import List

from .base_dataset import BaseDataset


class PascalVOCDataset(BaseDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])


def get_config(crop_size=(512, 512), batch_size=2, num_workers=8,
               channel_order='rgb',
               mean=(127.5, 127.5, 127.5),
               std=(63.75, 63.75, 63.75),
               scribble_dir='/home/yansu/dataset-scribble/data/Scribbles4All/s4Pascal/scribble/VOCScribble_aug_auto/',
               raw=False, ):
    train_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=255, ignore_idx=254, dir=scribble_dir, ),
        dict(module='scripts.dataset.pipeline.RandomResize',
             scale_size=(0.8, 1.2), aspect_ratio=(0.9, 1 / 0.9)),
        dict(module='scripts.dataset.pipeline.RandomCrop', crop_size=crop_size,
             cat_max_ratio=1.0, ignore_index=[255, -1], gt_name='gt_seg_map_scribble'),
        dict(module='scripts.dataset.pipeline.RandomFlip', prob=[1 / 4] * 3, direction=['h', 'v', 'd']),
        dict(module='scripts.dataset.pipeline.PhotoMetricDistortion', channel_order=channel_order,
             brightness_delta=10, contrast_range=(0.9, 1.1), saturation_range=(0.9, 1.1), hue_delta=5, ),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),

        dict(module='scripts.dataset.pipeline.Pad', size=crop_size, pad_val=0, ignore_idx=-1, random=True),
    ] if raw == False else [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=255, ignore_idx=254, dir=scribble_dir, ),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),    ]

    val_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
        dict(module='scripts.dataset.pipeline.Pad', size_divisor=64, pad_val=0, ignore_idx=-1, random=False),
    ]
    train_dataset = [dict(module='scripts.dataset.voc2012_aug.PascalVOCDataset',
                          data_root='/home/yansu/dataset-scribble/data/VOCdevkit/VOC2012',
                          img_path_prefix='JPEGImages',
                          seg_map_path_prefix='SegmentationClass',
                          ann_file='ImageSets/Segmentation/train.txt',
                          pipeline=train_pipeline, ),
                     dict(module='scripts.dataset.voc2012_aug.PascalVOCDataset',
                          data_root='/home/yansu/dataset-scribble/data/VOCdevkit/VOC2012',
                          img_path_prefix='JPEGImages',
                          seg_map_path_prefix='SegmentationClassAug',
                          ann_file='ImageSets/Segmentation/aug.txt',
                          pipeline=train_pipeline, ),
                     ]
    val_dataset = dict(module='scripts.dataset.voc2012_aug.PascalVOCDataset',
                       data_root='/home/yansu/dataset-scribble/data/VOCdevkit/VOC2012',
                       img_path_prefix='JPEGImages',
                       seg_map_path_prefix='SegmentationClass',
                       ann_file='ImageSets/Segmentation/val.txt',
                       pipeline=val_pipeline, )
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=dict(module='scripts.dataset.base_data_module.collate_from_dict_scribble'),
    )

    val_dataloader = dict(
        dataset=val_dataset,
        collate_fn=dict(module='scripts.dataset.base_data_module.collate_from_dict'),
        batch_size=1, shuffle=False, num_workers=num_workers,
    )
    num_classes = 21
    return num_classes, train_pipeline, train_dataset, train_dataloader, val_pipeline, val_dataset, val_dataloader
