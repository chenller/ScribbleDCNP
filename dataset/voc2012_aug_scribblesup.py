# from pathlib import Path
# import lightning as L
# from PIL import Image
# import numpy as np
# from typing import List


import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch


def xml2dict(fp):
    tree = ET.parse(fp)  # 解析 XML 文件
    root = tree.getroot()  # 获取根节点
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    filename = root.find('.//filename').text
    # 获取 segmented
    segmented = int(root.find('.//segmented').text)

    polygons_XY = []
    polygons = root.findall('.//polygon')

    # 遍历每个 <polygon> 元素
    for polygon_index, polygon in enumerate(polygons):
        XY = []
        # 获取当前 <polygon> 下的所有 <point> 元素
        tag = polygon.find('tag').text
        points = polygon.findall('point')

        # 遍历每个 <point> 元素，获取其中的 <X> 和 <Y> 值
        for point_index, point in enumerate(points):
            x = int(point.find('X').text)
            y = int(point.find('Y').text)
            XY.append([x, y])
        polygons_XY.append({'xy': XY, 'tag': tag})
    return dict(filename=filename, width=width, height=height, polygons=polygons_XY, )


def load_info(dir):
    fps_xml = [i for i in Path(dir).glob('**/*.xml') if
               'demo' not in str(i)]
    # fps_xml = fps_xml[:10]
    # 存储文件信息的字典
    infos = {}
    with ThreadPoolExecutor() as executor:
        # 使用 tqdm 来显示进度条
        for info in tqdm(executor.map(xml2dict, fps_xml), total=len(fps_xml)):
            filename = info.pop('filename')
            infos[filename] = info
    import json
    json.dump(infos, open(Path(dir) / 'info.json', 'w'))

    return info


def generate_mask(dir='/home/yansu/dataset-scribble/data/scribble_annotation/', info_filepath=None):
    if info_filepath is None:
        infos = load_info(dir)
    else:
        import json
        infos = json.load(open(info_filepath))

    tag_set = set()

    for key, info in infos.items():
        w, h = info['width'], info['height']
        polygons: list = info['polygons']
        for polygon in polygons:
            tag = polygon['tag']
            tag_set.add(tag)
    tag_list = list(tag_set)
    tag_list.sort()
    tag2num = {tag: i for i, tag in enumerate(tag_list)}

    save_path = Path(dir) / 'scribbles'
    save_path.mkdir(parents=True, exist_ok=True)
    # 遍历信息并生成图像
    for filename, info in tqdm(infos.items()):
        # 文件路径和大小
        filepath = (save_path / filename).with_suffix('.png')
        w, h = info['width'], info['height']

        # 创建空白灰度图
        img = Image.new('L', (w, h), color=255)  # 'L' 是灰度模式，背景设为黑色 (0)
        draw = ImageDraw.Draw(img)

        # 遍历每个多边形并绘制
        polygons = info['polygons']
        for polygon in polygons:
            tag = polygon['tag']
            num = tag2num.get(tag, 255)  # 使用默认值255，如果标签不存在于映射中
            xy = polygon['xy']  # 获取多边形的坐标
            # 绘制每一条边
            for i in range(len(xy) - 1):  # 不需要闭合
                start = xy[i]
                end = xy[i + 1]
                draw.line([start[0], start[1], end[0], end[1]], fill=num, width=5)  # 设置线宽为3

        # 保存图像
        img.save(filepath)


def get_config(crop_size=(512, 512), batch_size=2, num_workers=8,
               channel_order='rgb',
               mean=(127.5, 127.5, 127.5),
               std=(63.75, 63.75, 63.75),
               scribble_dir='/home/yansu/dataset-scribble/data/scribble_annotation/scribbles/',
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
        dict(module='scripts.dataset.pipeline.Normalize', mean=mean, std=std, ),
    ]

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
