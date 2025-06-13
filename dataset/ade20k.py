from .base_dataset import BaseDataset


class ADE20kDataset(BaseDataset):
    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])


def get_config(crop_size=(512, 512), batch_size=2, num_workers=8,
               channel_order='rgb',
               mean=(127.5, 127.5, 127.5),
               std=(63.75, 63.75, 63.75),
               scribble_dir='/home/yansu/dataset-scribble/data/Scribbles4All/s4ADE20K/scribbles/training/', raw=False):
    # train_pipeline = [
    #     # label : [0, num_classes -1], unlabel : 255, Pad : -1
    #     dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
    #     dict(module='scripts.dataset.pipeline.LoadAnnotations'),
    #     dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble', invalid_index=0, ignore_idx=255,
    #          dir='/home/yansu/dataset-scribble/data/Scribbles4All/s4ADE20K/scribbles/training/', ),
    #     dict(module='scripts.dataset.pipeline.ReduceZeroLabel',
    #          keys=('gt_seg_map', 'gt_seg_map_scribble'), ignore_idx=255),
    #     dict(module='scripts.dataset.pipeline.RandomResizeCrop',
    #          crop_size=crop_size, scale_size=(0.5, 2.0), aspect_ratio=(0.9, 1 / 0.9)),
    #     dict(module='scripts.dataset.pipeline.RandomFlip', prob=[0.5], direction=['horizontal']),
    #     dict(module='scripts.dataset.pipeline.PhotoMetricDistortion', channel_order=channel_order,
    #          brightness_delta=16, contrast_range=(0.75, 1.25), saturation_range=(0.75, 1.25), hue_delta=9, ),
    #     dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
    #
    #     dict(module='scripts.dataset.pipeline.Pad',
    #          size=crop_size, pad_val=0, ignore_idx=-1, random=True),
    # ]

    train_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.ReduceZeroLabel',
             keys=['gt_seg_map'], ignore_idx=255),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=0, ignore_idx=254, dir=scribble_dir, ),

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
    ] if raw == False else [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.ReduceZeroLabel',
             keys=['gt_seg_map'], ignore_idx=255),
        dict(module='scripts.dataset.pipeline.LoadAnnotationsScribble',
             invalid_index=0, ignore_idx=254, dir=scribble_dir, ),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
    ]

    val_pipeline = [
        # label : [0, num_classes -1], unlabel : 255, Pad : -1
        dict(module='scripts.dataset.pipeline.LoadImageFromFile', channel_order=channel_order, ),
        dict(module='scripts.dataset.pipeline.LoadAnnotations'),
        dict(module='scripts.dataset.pipeline.ReduceZeroLabel',
             keys=('gt_seg_map', 'gt_seg_map_scribble'), ignore_idx=255),
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
        dict(module='scripts.dataset.pipeline.Pad', size_divisor=64, pad_val=0, ignore_idx=-1, random=False),
    ]
    train_dataset = dict(module='scripts.dataset.ade20k.ADE20kDataset',
                         data_root='/home/yansu/dataset-scribble/data/ADEChallengeData2016/',
                         img_path_prefix='images/training/', seg_map_path_prefix='annotations/training/',
                         img_suffix='.jpg', seg_map_suffix='.png',
                         pipeline=train_pipeline, )
    val_dataset = dict(module='scripts.dataset.ade20k.ADE20kDataset',
                       data_root='/home/yansu/dataset-scribble/data/ADEChallengeData2016/',
                       img_path_prefix='images/validation/', seg_map_path_prefix='annotations/validation/',
                       img_suffix='.jpg', seg_map_suffix='.png',
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
    num_classes = 150
    return num_classes, train_pipeline, train_dataset, train_dataloader, val_pipeline, val_dataset, val_dataloader


if __name__ == '__main__':
    print(len(ADE20kDataset.METAINFO['classes']))
