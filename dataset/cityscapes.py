from pathlib import Path

import cv2
import numpy as np

from .base_dataset import BaseDataset


def convert_scribbles_into_labels(gtfine_labels_dir, scribbles_labels_dir):
    """
    Converts scribbled labels into more complete semantic segmentation labels using an existing fine label directory as reference.

    Args:
    - gtfine_labels_dir (str): Directory path containing ground truth labels in 'labelTrainIds' format.
    - scribbles_labels_dir (str): Directory path containing scribble annotations in 'labelIds' format.
    """
    # Load TrainID labels
    mmseg_gt = {p.stem[:-21]: p for p in Path(gtfine_labels_dir).glob('**/*_gtFine_labelTrainIds.png')}

    # Load scribble annotations
    scribbles_gt = {p.stem[:-16]: p for p in Path(scribbles_labels_dir).glob('**/*_gtFine_labelIds.png')}

    # Process each scribbled file
    for name, scr_gt_fp in scribbles_gt.items():
        # Read scribble and reference label images
        scr_gt = cv2.imread(str(scr_gt_fp), cv2.IMREAD_GRAYSCALE)
        gt_fp = mmseg_gt[name]
        gt = cv2.imread(str(gt_fp), cv2.IMREAD_GRAYSCALE)

        # Create a new array where to apply the scribble mask
        new_scr_gt = np.zeros_like(scr_gt, dtype=np.uint8)
        new_scr_gt[:] = 255
        mask = scr_gt != 0
        new_scr_gt[mask] = gt[mask]

        # Save the new label image with adjusted TrainIDs
        new_scr_gt_fp = scr_gt_fp.parent / gt_fp.name
        cv2.imwrite(str(new_scr_gt_fp), new_scr_gt)


# Sample usage
# convert_scribbles_into_labels('~/dataset/cityscapes/gtFine',
#                               '~/Scribbles4All/s4Cityscapes/scribbles/gtFine')


class CityscapesDataset(BaseDataset):
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


def get_config(crop_size=(512, 512), batch_size=2, num_workers=8,
               channel_order='rgb',
               mean=(127.5, 127.5, 127.5),
               std=(63.75, 63.75, 63.75),
               scribble_dir='/home/yansu/dataset-scribble/data/Scribbles4All/s4Cityscapes/scribbles/gtFine/train/',
               raw=False):
    train_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order),
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
        dict(module='scripts.dataset.pipeline.Pad',
             size=crop_size, pad_val=0, ignore_idx=-1, random=True),
    ] if raw==False else [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=255, ignore_idx=254, dir=scribble_dir, ),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
    ]

    val_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
        dict(module='scripts.dataset.pipeline.Pad',
             size_divisor=64, pad_val=0, ignore_idx=-1, random=False),
    ]
    train_dataset = dict(module='scripts.dataset.cityscapes.CityscapesDataset',
                         data_root='/home/yansu/dataset-scribble/data/cityscapes',
                         img_path_prefix='leftImg8bit/train', seg_map_path_prefix='gtFine/train',
                         img_suffix='_leftImg8bit.png', seg_map_suffix='_gtFine_labelTrainIds.png',
                         pipeline=train_pipeline, )
    val_dataset = dict(module='scripts.dataset.cityscapes.CityscapesDataset',
                       data_root='/home/yansu/dataset-scribble/data/cityscapes',
                       img_path_prefix='leftImg8bit/val', seg_map_path_prefix='gtFine/val',
                       img_suffix='_leftImg8bit.png', seg_map_suffix='_gtFine_labelTrainIds.png',
                       pipeline=val_pipeline, )
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True,
        collate_fn=dict(module='scripts.dataset.base_data_module.collate_from_dict_scribble'),
    )
    val_dataloader = dict(
        dataset=val_dataset,
        collate_fn=dict(module='scripts.dataset.base_data_module.collate_from_dict'),
        batch_size=1, shuffle=False, num_workers=num_workers,
    )
    num_classes = 19
    return num_classes, train_pipeline, train_dataset, train_dataloader, val_pipeline, val_dataset, val_dataloader


if __name__ == '__main__':
    print(len(CityscapesDataset.METAINFO['classes']))
