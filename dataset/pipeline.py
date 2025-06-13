import math
import time
from multiprocessing import Manager, RLock
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple, List, Iterable, Sequence
import cv2
from cachetools import LRUCache, cached, Cache
from sys import getsizeof
from PIL import Image
from io import BytesIO
import numpy as np
from abc import ABCMeta, abstractmethod

from turbojpeg import TurboJPEG, TJPF_BGR  # pip install PyTurboJPEG

from .. import dynamic_import
from threading import Lock

# 在父进程中创建共享缓存
# manager = Manager()
# global_cache_dict = manager.dict()  # 或者使用其他支持LRU的共享结构
# global_cache_lock = RLock()
#
#
# class SharedLRUCache(LRUCache):
#     def __init__(self, maxsize, getsizeof=None):
#         super().__init__(maxsize, getsizeof)
#         self.__order = global_cache_dict
#
#     def __getitem__(self, *args, **kwargs):
#         with global_cache_lock:
#             return super().__getitem__(*args, **kwargs)
#
#     def __setitem__(self, *args, **kwargs):
#         with global_cache_lock:
#             return super().__setitem__(*args, **kwargs)
#
#     def __delitem__(self, *args, **kwargs):
#         with global_cache_lock:
#             return super().__delitem__(*args, **kwargs)
#
#     def __update(self, *args, **kwargs):
#         with global_cache_lock:
#             return super().__update(*args, **kwargs)
#
#     def popitem(self):
#         with global_cache_lock:
#             return super().popitem()
#
#
# global_cache = SharedLRUCache(maxsize=5 * 1024 ** 3)


# Initialize a 10GB LRU Cache for storing image bytes
global_cache = LRUCache(maxsize=1.5 * 1024 ** 3, getsizeof=getsizeof)


def compose(pipeline):
    pipeline_obj = []
    for p in pipeline:
        assert isinstance(p, dict), "pipeline must be a dict"
        pipeline_obj.append(dynamic_import(**p))
    return pipeline_obj


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """


def _decode_image_cv2(img_bytes: bytes, channel_order='bgr') -> np.ndarray:
    """Decode image using OpenCV."""
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if channel_order == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _decode_image_pillow(img_bytes: bytes, channel_order='rgb') -> np.ndarray:
    """Decode image using Pillow."""
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    if channel_order == 'bgr':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


JPEG = TurboJPEG()


def _decode_image_TurboJPEG(img_bytes: bytes, channel_order='rgb') -> np.ndarray:
    """Decode image using OpenCV."""
    img = JPEG.decode(img_bytes, pixel_format=TJPF_BGR)
    if channel_order == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@cached(global_cache)
def read_img_cache_array(filename: str, channel_order='rgb', imdecode_backend='cv2') -> np.ndarray:
    with open(filename, 'rb') as f:
        img_bytes = f.read()
    if str(filename).endswith('.jpg'):
        img = _decode_image_TurboJPEG(img_bytes, channel_order=channel_order)
    elif imdecode_backend == 'cv2':
        img = _decode_image_cv2(img_bytes, channel_order=channel_order)
    elif imdecode_backend == 'pillow':
        img = _decode_image_pillow(img_bytes, channel_order=channel_order)
    else:
        raise ValueError("Unknown image decode")
    return img


@cached(global_cache)
def read_bytes(filename: str) -> bytes:
    """Cached function to read image bytes from file."""
    with open(filename, 'rb') as f:
        bytes = f.read()
    return bytes


def read_img_cache_bytes(filename: str, channel_order='rgb', imdecode_backend='cv2') -> np.ndarray:
    img_bytes = read_bytes(filename)
    if str(filename).endswith('.jpg'):
        img = _decode_image_TurboJPEG(img_bytes, channel_order=channel_order)
    elif imdecode_backend == 'cv2':
        img = _decode_image_cv2(img_bytes, channel_order=channel_order)
    elif imdecode_backend == 'pillow':
        img = _decode_image_pillow(img_bytes, channel_order=channel_order)
    else:
        raise ValueError("Unknown image decode")
    return img


class LoadImageFromFile(BaseTransform):
    def __init__(self, channel_order='rgb', imdecode_backend='cv2', cache='bytes'):
        assert channel_order in ['rgb', 'bgr'], "channel_order must be 'rgb' or 'bgr'"
        assert imdecode_backend in ['pillow', 'cv2'], "imdecode_backend must be 'pillow' or 'cv2'"
        self.channel_order = channel_order
        self.imdecode_backend = imdecode_backend
        self.cache = cache

    def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        filename = results['img_path']
        if self.cache == 'bytes':
            img = read_img_cache_bytes(str(filename), channel_order=self.channel_order,
                                       imdecode_backend=self.imdecode_backend)
        else:
            img = read_img_cache_array(str(filename), channel_order=self.channel_order,
                                       imdecode_backend=self.imdecode_backend)
        # print(id(global_cache))
        results['img'] = img
        results['raw_img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class LoadAnnotations(BaseTransform):
    def transform(self, results: Dict) -> Optional[Dict[str, Any]]:
        img_bytes = read_bytes(results['seg_map_path'])
        # img=_decode_image_cv2(img_bytes)
        # img=img[:,:,0]
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img, dtype=int)
        results['raw_gt_seg_map'] = img
        results['gt_seg_map'] = img
        results['seg_fields'].append('gt_seg_map')
        return results


class LoadAnnotationsScribble(BaseTransform):
    def __init__(self, dir, invalid_index, ignore_idx=254, key_name='gt_seg_map_scribble',
                 replace: List[str] = None, key_num_part=1):
        self.dir = Path(dir)
        self.invalid_index = invalid_index
        self.ignore_idx = ignore_idx
        self.key_name = key_name
        self.replace = replace
        self.key_num_part = key_num_part
        self.key2path = {}
        for fp in self.dir.glob('**/*.png'):
            fp: Path
            key = '/'.join(fp.parts[-key_num_part:])
            self.key2path[key] = fp

    def transform(self, results: Dict) -> Optional[Dict[str, Any]]:
        key = '/'.join(Path(results['seg_map_path']).parts[-self.key_num_part:])
        if self.replace is not None:
            key = key.replace(*self.replace)
        scribble_fp = self.key2path[key]
        img_bytes = read_bytes(str(scribble_fp))
        # img = _decode_image_cv2(img_bytes)
        # img = img[:, :, 0]
        array = Image.open(BytesIO(img_bytes))
        array = np.array(array)
        if array.ndim == 3:
            array = array[..., 0]
        results['raw_' + self.key_name] = np.copy(array)
        valid_index = (array != self.invalid_index)
        gt = results['gt_seg_map']
        array[~valid_index] = self.ignore_idx
        array[valid_index] = gt[valid_index]
        array[gt == 255] = 255
        array = array.astype(int)
        # img[img < 0] = 255
        # print(np.unique(img))
        results['scribble_path'] = str(scribble_fp)
        results[self.key_name] = array
        results['seg_fields'].append(self.key_name)
        return results


class ReduceZeroLabel(BaseTransform):
    def __init__(self, keys=['gt_seg_map'], ignore_idx=255):
        self.keys = keys
        self.ignore_idx = ignore_idx

    def transform(self, results: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        reduce_zero_label_info = []
        for key in self.keys:
            array = results.get(key, None)
            if array is None:
                continue
            array[array == 0] = self.ignore_idx
            array = array - 1
            array[array == self.ignore_idx - 1] = self.ignore_idx
            results[key] = array
            reduce_zero_label_info.append(key)
        results['reduce_zero_label'] = reduce_zero_label_info
        return results


class Normalize(BaseTransform):
    def __init__(self, mean, std, ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def transform(self, results: Dict) -> Optional[Dict[str, Any]]:
        img = results['img']
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        results['img'] = img
        results['mean'] = self.mean
        results['std'] = self.std
        return results


class Pad(BaseTransform):
    def __init__(self, size: Tuple[int, int] = None, size_divisor: int = None, pad_val=0, ignore_idx=255, random=False):
        """
        Initialize the transform with specified target size and padding value.

        :param size: Target size (height, width).
        :param pad_val: Value to use for padding. Default is 0.
        """
        assert not ((size is None) and (size_divisor is None)), \
            f"Only one of 'size' or 'size_divisor' should be specified. but got {size=} and {size_divisor=}"
        assert not ((size is not None) and (size_divisor is not None)), \
            f"Either 'size' or 'size_divisor' must be specified. but got {size=} and {size_divisor=}"
        if size is not None:
            assert len(size) == 2
            assert isinstance(size[0], int) and isinstance(size[1], int)
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.ignore_idx = ignore_idx
        self.random = random

    def get_pad_shape(self, shape: Tuple[int, int]):
        h, w = shape
        if self.size is not None:
            target_h, target_w = self.size
        else:
            target_h = math.ceil(h / self.size_divisor) * self.size_divisor
            target_w = math.ceil(w / self.size_divisor) * self.size_divisor

        if h >= target_h and w >= target_w:  # No need to pad if the image is already larger than the target size
            return 0, 0, 0, 0, dict(top=0, bottom=0, left=0, right=0, crop_slice=(slice(0, h), slice(0, w)))

        # Calculate padding sizes
        pad_h = target_h - h
        pad_w = target_w - w
        # top_pad = max(0, (target_h - h) // 2)
        # left_pad = max(0, (target_w - w) // 2)
        if self.random:
            top_pad = np.random.randint(0, pad_h + 1)
            left_pad = np.random.randint(0, pad_w + 1)
        else:
            top_pad = 0
            left_pad = 0
        bottom_pad = max(0, target_h - h - top_pad)
        right_pad = max(0, target_w - w - left_pad)
        crop_slice = (slice(top_pad, top_pad + h), slice(left_pad, left_pad + w))
        return (top_pad, left_pad, bottom_pad, right_pad,
                dict(top=top_pad, bottom=bottom_pad, left=left_pad, right=right_pad, crop_slice=crop_slice))

    def pad(self, img: np.ndarray, top_pad, left_pad, bottom_pad, right_pad, pad_val):
        # if img.dtype==np.uint8:
        #     img = img.astype(int)
        if len(img.shape) == 3:  # For RGB/BGR images
            padded_img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                                mode='constant', constant_values=pad_val)
        else:  # For grayscale images
            padded_img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)),
                                mode='constant', constant_values=pad_val)
        return padded_img

    def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        img = results['img']
        top_pad, left_pad, bottom_pad, right_pad, pad_info = self.get_pad_shape(img.shape[:2])
        img = self.pad(img, top_pad, left_pad, bottom_pad, right_pad, self.pad_val)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_info'] = pad_info

        # Apply padding to segmentation fields if available
        for key in results.get('seg_fields', []):
            results[key] = self.pad(results[key], top_pad, left_pad, bottom_pad, right_pad, self.ignore_idx)
        return results


class RandomFlip(BaseTransform):
    def __init__(self, prob: Iterable[float] = [0.5],
                 direction: Sequence[Optional[str]] = ['horizontal']):
        assert len(prob) == len(direction), "prob and direction must have same length"
        self.prob = prob
        self.direction = direction

        valid_directions = ['horizontal', 'vertical', 'diagonal', 'h', 'v', 'd']
        assert all([d in valid_directions for d in direction]), "direction must be one of {}".format(valid_directions)
        self.cum_prob = np.cumsum(prob)
        assert self.cum_prob[-1] <= 1, "prob must sum to less than or equal to 1"

    def flip_img(self, img: np.ndarray, direction: str) -> np.ndarray:
        if direction in ['horizontal', 'h']:
            return np.fliplr(img)
        elif direction in ['vertical', 'v']:
            return np.flipud(img)
        elif direction in ['diagonal', 'd']:
            return np.flip(img, (0, 1))
        else:
            raise ValueError("Unsupported flip direction")

    def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Functions to randomly flip the image and segmentation maps.

        Args:
            results (dict): Result dict from dataset.

        Returns:
            dict: The dict contains flipped image and meta information.
        """
        img = results['img']
        random_prob = np.random.rand()
        for p, d in zip(self.cum_prob, self.direction):
            if random_prob < p:
                img = self.flip_img(img, d)
                # Apply flipping to segmentation fields if available
                for key in results.get('seg_fields', []):
                    results[key] = self.flip_img(results[key], d)
                results['flip_direction'] = d
        results['img'] = img
        results['img_shape'] = img.shape
        return results


class RandomResize(BaseTransform):
    """
    对给定的图像进行缩放
    参数:
    - scale_size: 图像的整体缩放比例。
    - aspect_ratio: 宽高比 (width/height)。
    """

    def __init__(self, scale_size: Tuple[float, float] = (0.5, 2.0),
                 aspect_ratio: Tuple[float, float] = (0.9, 1 / 0.9)):
        self.scale_size = scale_size
        self.aspect_ratio = aspect_ratio

    def transform(self, results: dict) -> dict:
        img = results['img']

        # 随机选择一个浮点数作为缩放比例
        scale_factor = np.random.uniform(self.scale_size[0], self.scale_size[1])
        # 随机选择一个浮点数作为宽高比
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        height, width = img.shape[:2]
        # 计算新的宽度和高度
        new_height = int(height * scale_factor / np.sqrt(aspect_ratio))
        new_width = int(width * scale_factor * np.sqrt(aspect_ratio))
        # 缩放图像
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = cv2.resize(results[key], (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # 更新结果
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale_factor'] = scale_factor
        results['aspect_ratio'] = aspect_ratio

        return results


class RandomCrop(BaseTransform):
    """Random crop the image & seg.
    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 gt_name='gt_seg_map',
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
                isinstance(crop_size, tuple | list) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        if isinstance(ignore_index, tuple) or isinstance(ignore_index, list):
            self.ignore_index = ignore_index
        else:
            self.ignore_index = [ignore_index]
        self.gt_name = gt_name

    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results[self.gt_name], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                for idx in self.ignore_index:
                    cnt = cnt[labels != idx]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class RandomResizeCrop(BaseTransform):
    def __init__(self, crop_size=(512, 512),
                 scale_size: Tuple[float, float] = (0.5, 2.0),
                 aspect_ratio: Tuple[float, float] = (0.9, 1 / 0.9)):
        self.crop_size = crop_size  # (h,w)
        self.scale_size = scale_size
        self.aspect_ratio = aspect_ratio
        self._random_crop_obj = RandomCrop(crop_size, cat_max_ratio=1.)

    def crop_size_with_resize(self):
        h, w = self.crop_size
        # 随机选择一个浮点数作为缩放比例
        scale_factor = np.random.uniform(self.scale_size[0], self.scale_size[1])
        # 随机选择一个浮点数作为宽高比
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        # scale_factor = 0.68
        # print(scale_factor, aspect_ratio)
        # 计算新的宽度和高度
        new_height = int(h / scale_factor * np.sqrt(aspect_ratio))
        new_width = int(w / scale_factor / np.sqrt(aspect_ratio))
        return scale_factor, aspect_ratio, (new_height, new_width)

    def generate_crop_bbox(self, img: np.ndarray, h, w) -> Tuple[int]:
        """Randomly get a crop bounding box.

        Args:
            img (np.ndarray): Original input image.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        margin_h = max(img.shape[0] - h, 0)
        margin_w = max(img.shape[1] - w, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + h
        crop_x1, crop_x2 = offset_w, offset_w + w
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img: np.ndarray,
             crop_bbox: tuple,
             crop_size_new: tuple,
             interpolation=cv2.INTER_NEAREST) -> np.ndarray:

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]), interpolation=interpolation)

        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = img.shape[:2]
        scale_factor, aspect_ratio, crop_size_new = self.crop_size_with_resize()
        if crop_size_new[0] > h or crop_size_new[1] > w:
            height, width = img.shape[:2]
            # 计算新的宽度和高度
            new_height = int(height * scale_factor / np.sqrt(aspect_ratio))
            new_width = int(width * scale_factor * np.sqrt(aspect_ratio))
            # 缩放图像
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            # crop semantic seg
            for key in results.get('seg_fields', []):
                results[key] = cv2.resize(results[key], (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            results = self._random_crop_obj.transform(results)
        else:
            crop_bbox = self.generate_crop_bbox(img, *crop_size_new)

            img = self.crop(img, crop_bbox, crop_size_new, interpolation=cv2.INTER_LINEAR)

            # crop semantic seg
            for key in results.get('seg_fields', []):
                results[key] = self.crop(results[key], crop_bbox, crop_size_new, interpolation=cv2.INTER_NEAREST)

            results['img'] = img
            results['img_shape'] = img.shape[:2]
        results['scale_factor'] = scale_factor
        results['aspect_ratio'] = aspect_ratio

        return results


class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18,
                 channel_order='rgb'):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        if contrast_range is not None:
            assert len(contrast_range) == 2, 'contrast_range must be a tuple of length 2'
            self.contrast_lower, self.contrast_upper = contrast_range
        if saturation_range is not None:
            assert len(saturation_range) == 2, 'saturation_range must be a tuple of length 2'
            self.saturation_lower, self.saturation_upper = saturation_range

        self.channel_order = channel_order.upper()
        assert self.channel_order in ['RGB', 'BGR'], 'channel_order must be RGB or BGR'

    def convert(self, img: np.ndarray, alpha: int | float = 1, beta: int | float = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if np.random.randint(2):
            return self.convert(img, beta=np.random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if np.random.randint(2):
            return self.convert(img, alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if np.random.randint(2):
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_{self.channel_order.upper()}2HSV'))
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=np.random.uniform(self.saturation_lower,
                                                                              self.saturation_upper))
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_HSV2{self.channel_order.upper()}'))

        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """
        if np.random.randint(2):
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_{self.channel_order.upper()}2HSV'))
            img[:, :, 0] = (img[:, :, 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_HSV2{self.channel_order.upper()}'))
        return img

    def saturation_and_hue(self, img: np.ndarray) -> np.ndarray:
        is_saturation = np.random.randint(2)
        is_hue = np.random.randint(2)
        if (is_saturation or is_hue) and ((self.saturation_range is not None) or (self.hue_delta is not None)):
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_{self.channel_order.upper()}2HSV'))
            if is_saturation and self.saturation_range is not None:
                img[:, :, 1] = self.convert(img[:, :, 1], alpha=np.random.uniform(self.saturation_lower,
                                                                                  self.saturation_upper))
            if is_hue and (self.hue_delta is not None):
                img[:, :, 0] = (img[:, :, 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, getattr(cv2, f'COLOR_HSV2{self.channel_order.upper()}'))
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1 and self.contrast_range is not None:
            img = self.contrast(img)

        # # random saturation
        # img = self.saturation(img)
        #
        # # random hue
        # img = self.hue(img)
        img = self.saturation_and_hue(img)

        # random contrast
        if mode == 0 and self.contrast_range is not None:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
