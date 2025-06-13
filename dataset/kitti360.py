from pathlib import Path

import cv2
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from .base_dataset import BaseDataset


def convert_semantic_to_trainid(dir='/home/yansu/dataset-scribble/data/KITTI360/data_2d_semantics/'):
    def labelmap(fp, id2trainId: dict[int, int], color=None):
        gt = cv2.imread(str(fp))[:, :, 0]
        gtmap = np.full_like(gt, fill_value=255, dtype=np.uint8)
        for id, trainid in id2trainId.items():
            gtmap[gt == id] = trainid

        gtmappil = Image.fromarray(gtmap)
        gtmappil.putpalette(color)
        save_path = Path(str(fp).replace('/semantic/', '/semantic_trainId/'))
        save_path.parent.mkdir(exist_ok=True, parents=True)
        gtmappil.save(str(save_path))

    def process_wrapper(args):
        return labelmap(*args)  # Unpack the tuple here

    id2trainId = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                  27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    palette = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
               (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
               (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    palette_use = []
    for p in palette:
        palette_use.append(p[0])
        palette_use.append(p[1])
        palette_use.append(p[2])
    palette_use += [255] * (256 * 3 - len(palette_use))
    palette_use = tuple(palette_use)
    dir = Path(dir)
    gtfps = [i for i in dir.glob('**/*.png') if '/semantic/' in str(i)]

    # Prepare arguments for multiprocessing
    args_list = [(gtfp, id2trainId, palette_use) for gtfp in gtfps]

    # Use multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_wrapper, args_list), total=len(gtfps)))

    print(len(gtfps))


class KITTI360Dataset(BaseDataset):
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
                 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'),
        palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    )

    def __init__(self, data_root, ann_file=None, pipeline: list = None):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline)

    def init(self):
        data_root = Path(self.data_root)
        assert self.ann_file is not None
        info_path = data_root / self.ann_file
        with open(str(info_path), 'r') as f:
            infos = f.readlines()
        for info in infos:
            info = info.removesuffix('\n').split(' ')

            img_fp, seg_map_fp = info
            img_fp = data_root / img_fp
            seg_map_fp = data_root / seg_map_fp
            seg_map_fp = str(seg_map_fp).replace('/semantic/', '/semantic_trainId/')
            self.data_list.append({'img_path': str(img_fp), 'seg_map_path': str(seg_map_fp),
                                   'seg_fields': [], 'reduce_zero_label': None})


def get_config(crop_size=(352, 1408), batch_size=2, num_workers=8,
               channel_order='rgb',
               mean=(127.5, 127.5, 127.5),
               std=(63.75, 63.75, 63.75),
               scribble_dir='/home/yansu/dataset-scribble/data/Scribbles4All/s4KITTI360',
               raw=False):
    train_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=0, ignore_idx=254, dir=scribble_dir, key_num_part=4,
             replace=['/semantic_trainId/', '/scribble/']),
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
    ]if raw==False else[
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=0, ignore_idx=254, dir=scribble_dir, key_num_part=4,
             replace=['/semantic_trainId/', '/scribble/']),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
    ]

    val_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
        dict(module='scripts.dataset.pipeline.Pad',
             size_divisor=32, pad_val=0, ignore_idx=-1, random=False),
    ]
    train_dataset = dict(module='scripts.dataset.kitti360.KITTI360Dataset',
                         data_root='/home/yansu/dataset-scribble/data/KITTI360/',
                         ann_file='data_2d_semantics/train/2013_05_28_drive_train_frames.txt',
                         pipeline=train_pipeline, )
    val_dataset = dict(module='scripts.dataset.kitti360.KITTI360Dataset',
                       data_root='/home/yansu/dataset-scribble/data/KITTI360/',
                       ann_file='data_2d_semantics/train/2013_05_28_drive_val_frames.txt',
                       pipeline=val_pipeline, )
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
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
    print(len(KITTI360Dataset.METAINFO['classes']))
