# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Dataloaders and dataset utils."""
import warnings
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
import nibabel as nib
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

import torchio as tio
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
from scipy.ndimage import label
from typing import Dict, Sequence, Union, Tuple, Optional

from torchio.transforms import SpatialTransform

from utils.nnunet.paths import nnUNet_preprocessed
from utils.nnunet.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from utils.augmentations import (
    Albumentations,
    augment_hsv,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    cv2,
    segments2boxes,
    unzip_file,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Inherit from DistributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
):
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


def seg_to_instances(
    seg: np.ndarray,
    min_num_voxel: int = 0,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Use connected components with ones matrix to create
    instances from segmentation

    Args:
        seg: semantic segmentation [spatial dims]
        min_num_voxel: minimum number of voxels of an instance

    Returns:
        np.ndarray: instance segmentation
        Dict[int, int]: mapping from instances to classes
    """
    structure = np.ones([3] * seg.ndim)

    unique_classes = np.unique(seg)
    unique_classes = unique_classes[unique_classes > 0]

    instances = np.zeros_like(seg)
    instance_classes = {}

    i = 1
    for uc in unique_classes:
        binary_class_mask = (seg == uc)
        instances_temp, _ = label(binary_class_mask, structure=structure)

        instance_ids = np.unique(instances_temp)
        instance_ids = instance_ids[instance_ids > 0]

        for iid in instance_ids:
            instance_binary_mask = instances_temp == iid

            if min_num_voxel > 0:
                if instance_binary_mask.sum() < min_num_voxel:  # remove small instances
                    continue

            instances[instance_binary_mask] = i  # save instance to final mask
            instance_classes[int(i)] = uc
            i = i + 1  # bump instance index
    return instances, instance_classes


def get_bbox_np(seg: np.ndarray,
                map_dict: Optional[Dict[Union[str, int], Union[str, int]]] = None,
                **kwargs,
                ) -> dict:
    """
    Get bounding boxes and mapping from instances to classes
    
    Args:
        seg: instance segmentation [1, dims]
        mapping: define mapping from instance ids to classes
    
    Returns:
        dict: extracted boxes and classes
            `boxes` (np.ndarray): bounding boxes [N, dims * 2]
            `classes` (np.ndarray): classes (in same order as boxes) [N]
    """
    if map_dict is not None:
        map_dict = {str(key): str(item) for key, item in map_dict.items()}

    result = {}
    boxes, instance_idx = instances_to_boxes_np(seg[0], **kwargs)
    

    result["boxes"] = boxes

    # if map_dict is not None:
    #     box_classes = get_instance_class_from_properties_seq(instance_idx, map_dict)
    #     result["classes"] = np.array(box_classes)
    return result



def instances_to_boxes_np(instances: np.ndarray, max_det=20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bounding boxes from instance segmentation.

    Args:
        instances: instance segmentation [dims]

    Returns:
        np.ndarray: bounding boxes [1, 20, 2, 3]
        np.ndarray: instance indices [20]
    """
    instance_ids = np.unique(instances)
    instance_ids = instance_ids[instance_ids > 0]
    boxes = np.zeros((1, max_det, 2, 3), dtype=np.float32)
    instance_idx = np.zeros(max_det, dtype=np.int32)

    box_count = 0
    for instance_id in instance_ids:
        if box_count >= max_det:
            break
        instance_mask = (instances == instance_id)
        coords = np.array(np.nonzero(instance_mask))
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1)
        box = np.stack([min_coords[1:], max_coords[1:]], axis=0)
        boxes[0, box_count, :, :coords.shape[0]-1] = box
        instance_idx[box_count] = instance_id
        box_count += 1

    return boxes, instance_idx



def get_instance_class_from_properties_seq(instance_idx: np.ndarray, map_dict: Dict[str, str]) -> list:
    """
    Get instance class from properties sequence.

    Args:
        instance_idx: instance indices [N]
        map_dict: mapping from instance ids to classes

    Returns:
        list: classes (in same order as instance indices) [N]
    """
    return [int(map_dict[str(idx)]) for idx in instance_idx]


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)


class LoadImages:
    """YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes YOLOv5 loader for images/videos, supporting glob patterns, directories, and lists of paths."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self

    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Initializes a new video capture object with path, frame count adjusted by stride, and orientation
        metadata.
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on its orientation; supports 0, 90, and 180 degrees rotations."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files



class LoadImages3D:
    def __init__(self, paths, img_size=640, stride=32, auto=True, transforms=None):
        """Initializes loader for multiple 3D image datasets."""
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = "image"

      
        self.files =  sorted(glob.glob(os.path.join(paths,'*.nii.gz')))

        self.nf = len(self.files)  # Number of files
        if self.nf == 0:
            raise ValueError("No valid 3D image files found.")
        
        self.file_index = 0  # Index to track which file is being processed
        self.last_file_index = -1  # Track the last processed file index
        self.slices = []  # To store slices of the current file
        self.load_slices(self.files[self.file_index])  # Load slices from the first file

    def load_slices(self, path):
        """Load slices from a 3D data file."""
        img = nib.load(path)
        data = img.get_fdata()
        self.slices = [cv2.convertScaleAbs(data[:, :, i], alpha=(255.0/data[:, :, i].max())) for i in range(data.shape[2])]
        self.ns = len(self.slices)  # Update number of slices
        self.count = 0  # Reset slice counter
        self.last_file_index = self.file_index  # Update last file index after loading new slices

    def __iter__(self):
        return self

    def __next__(self):
        """Advances to the next slice, or the next file if all slices are processed."""
        if self.count == self.ns:
            self.file_index += 1
            if self.file_index == self.nf:
                raise StopIteration  # No more files
            self.load_slices(self.files[self.file_index])  # Load new file slices

        im0 = self.slices[self.count]
        ori_shape = im0.shape
        path = self.files[self.file_index]
        s = f"file {self.file_index + 1}/{self.nf}, slice {self.count + 1}/{self.ns} {path}: "

        im0 = np.expand_dims(im0,axis=2)
        im0 = np.repeat(im0,3,-1)
        
        if self.transforms:
            im = self.transforms(im0)  # Apply transformations
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)

        self.count += 1
        return path, im, im0, None, s, ori_shape

    def is_new_file_started(self):
        """Check if a new file has been started."""
        return self.last_file_index != self.file_index

    def get_slice_num(self):
        return self.ns
    
    def get_current_index(self):
        return self.count
    
    def is_last_slice(self):
        """Check if it is the last slice of the current file."""
        return self.count == self.ns

    def __len__(self):
        """Returns the total number of slices across all files."""
        return sum(self.ns for _ in self.files)  # Assume self.ns updated for each file

def map_data(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())

    # Map the normalized data to the range [0, 255] for display
    mapped_data = (normalized_data * 255).astype(np.uint8)
    
    return mapped_data

def slice_3d_data(data: np.ndarray):
    """slice 3D Data into 2D patches"""
    slices = [map_data(data[ :, :, i]) for i in range(data.shape[2])]
    # slices = [cv2.convertScaleAbs(data[:, :, i], alpha=(255.0/data[:, :, i].max())) for i in range(data.shape[2])]
    return slices

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class SliceDataset(Dataset):
    def __init__(self, image3D, gt3d):
        # import pdb; pdb.set_trace()
        self.slices = slice_3d_data(image3D.squeeze(0).squeeze(0))
        self.gt_slices = slice_3d_data(gt3d.squeeze(0).squeeze(0))
        self.img_size = 512
        self.stride = 32
        self.auto = True
        
    def __len__(self):
        return len(self.slices)
    
    def preposss_im(self, im):
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        
        return im

    def __getitem__(self, idx):
        slice_ = self.slices[idx]
        slice_gt = self.gt_slices[idx]
        ori_slice_, ori_slice_gt = slice_.copy(), slice_gt.copy()
        slice_, ratio, pad= letterbox(slice_, self.img_size, stride=self.stride, auto=self.auto)  # padded resize
        slice_gt = letterbox(slice_gt, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        
        slice_ = self.preposss_im(slice_)
        slice_gt = self.preposss_im(slice_gt)
        ori_slice_ = self.preposss_im(ori_slice_)
        ori_slice_gt = self.preposss_im(ori_slice_gt)
  
        return torch.from_numpy(slice_), torch.from_numpy(ori_slice_), torch.from_numpy(ori_slice_gt), torch.tensor(idx, dtype=torch.float32).unsqueeze(0), ratio, pad  # (1, 512, 512)

class Dataset_Union_ALL(Dataset, SpatialTransform): 
    def __init__(
        self,
        paths,
        dataset_id=130,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=500,
        split_num=1,
        split_idx=0,
        pcc=False,
        pcc_probability=0.3,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.target_shape=(self.image_size, self.image_size, self.image_size)
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info
        try:
            dataset_name = convert_id_to_dataset_name(dataset_id)
            with open(f'{nnUNet_preprocessed}/{dataset_name}/dataset_fingerprint.json') as f:
                dataset_fingerprint = json.load(f)
            self.intensityproperties = dataset_fingerprint['foreground_intensity_properties_per_channel']['0']
        except Exception as e:
            raise Exception(f"Error loading dataset_fingerprint from {f'{nnUNet_preprocessed}/{dataset_name}/dataset_fingerprint.json'}: {e}\n") from e
        
        self.target_dtype = np.float32
        
        self.mask_name = 'label'
        self.labels = [1]
        self.pcc_probability = pcc_probability
        
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])
        

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        sitk_image_arr, _ = sitk_to_nib(sitk_image)
        sitk_label_arr, _ = sitk_to_nib(sitk_label)
        
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        sitk_image_arr = sitk_image_arr.astype(self.target_dtype, copy=False)
        np.clip(sitk_image_arr, lower_bound, upper_bound, out=sitk_image_arr)
        sitk_image_arr -= mean_intensity
        sitk_image_arr /= max(std_intensity, 1e-8)
        
        # import pdb; pdb.set_trace()

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
            label=tio.LabelMap(tensor=sitk_label_arr),
        )
        
        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
        
        
        
        if self.mode == "train" and self.data_type == "Tr":
            
           
        

            if self.pcc and np.random.rand() < self.pcc_probability:
                # print("using pcc setting")
                # crop from random click point
                random_index = torch.argwhere(subject.label.data == 1)
                if len(random_index) >= 1:
                    random_index = random_index[np.random.randint(0, len(random_index))]
                    # print(random_index)
                    crop_mask = torch.zeros_like(subject.label.data)
                    # print(crop_mask.shape)
                    crop_mask[random_index[0]][random_index[1]][random_index[2]][
                        random_index[3]
                    ] = 1
                    subject.add_image(
                        tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                        image_name="crop_mask",
                    )
                    subject = tio.CropOrPad(
                        mask_name="crop_mask",
                        target_shape=(self.image_size, self.image_size, self.image_size),
                    )(subject)
            else:
                cropping_params, subject, subject_shape = self._compute_mask_center_crop_or_pad_with_Resize(subject)
                # 对 subject 进行随机反转
                random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
                subject = random_flip(subject)
            
        # Calculating the 3D outer box of an example
        label_data_np = subject.label.data.numpy()
        instances_not_filtered, instances_not_filtered_classes = seg_to_instances(label_data_np)
        
        final_mapping = {}
        if instances_not_filtered.max() > 0:
            boxes = get_bbox_np(instances_not_filtered[None])["boxes"]
            # print(boxes.shape)
        else:
            boxes = np.zeros((1,20,2,3))
        
        if self.mode == "train" and self.data_type == "Tr":
        
            return (
                subject.image.data.clone().detach().float(),
                subject.label.data.clone().detach().float(),
                boxes,
            )
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "origin": sitk_label.GetOrigin(),
                "direction": sitk_label.GetDirection(),
                "spacing": sitk_label.GetSpacing(),
            }
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                boxes,
                meta_info,
            )
        else:
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                boxes,
                self.image_paths[index],
            )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        # import pdb; pdb.set_trace()
        for path in paths:
            d = os.path.join(path, f"labels{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    label_path = os.path.join(
                        path, f"labels{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(label_path.replace("labels", "images"))
                    self.label_paths.append(label_path)
    
    @staticmethod
    def _bbox_mask(mask_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

        Args:
            mask_volume: 3D NumPy array.
        """  # noqa: B950
        i_any = np.any(mask_volume, axis=(1, 2))
        j_any = np.any(mask_volume, axis=(0, 2))
        k_any = np.any(mask_volume, axis=(0, 1))
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        k_min, k_max = np.where(k_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min, k_min])
        bb_max = np.array([i_max, j_max, k_max]) + 1
        return bb_min, bb_max
    
    def _compute_mask_center_crop_or_pad_with_Resize(self, subject):
        # Retrieve mask data
        mask_data = self.get_mask_from_masking_method(
            self.mask_name,
            subject,
            subject[self.mask_name].data,
            self.labels,
        ).numpy()

        subject_shape = subject.spatial_shape
        bb_min, bb_max = self._bbox_mask(mask_data[0])
        center_mask = np.mean((bb_min, bb_max), axis=0)
        padding = []
        cropping = []

        if self.target_shape is None:
            target_shape = bb_max - bb_min
        else:
            target_shape = self.target_shape

        for dim in range(3):
            target_dim = target_shape[dim]
            center_dim = center_mask[dim]
            
            center_on_index = not (center_dim % 1)
            target_even = not (target_dim % 2)

            # Approximation when the center cannot be computed exactly
            # The output will be off by half a voxel, but this is just an
            # implementation detail
            if target_even ^ center_on_index:
                center_dim -= 0.5
            
            subject_dim = subject_shape[dim]
            bb_dim = bb_max[dim] - bb_min[dim]
            
            if bb_dim < target_dim:
                # If the bounding box size is smaller than the target size,
                # we can directly crop to the target size
                crop_dim = target_dim
            else:
                # If the bounding box size is larger than the target size,
                # we need to crop based on the bounding box size first,
                # then resize to the target size
                crop_dim = bb_dim
            
            begin = center_dim - crop_dim / 2
            if begin >= 0:
                crop_ini = begin
                pad_ini = 0
            else:
                crop_ini = 0
                pad_ini = -begin

            end = center_dim + crop_dim / 2
            if end <= subject_dim:
                # Since tio.Crop uses subject's shape minus crop_fin, calculate it this way
                crop_fin = subject_dim - end
                pad_fin = 0
            else:
                crop_fin = 0
                pad_fin = end - subject_dim
                
            padding.extend([pad_ini, pad_fin])
            cropping.extend([crop_ini, crop_fin])

        padding_array = np.asarray(padding, dtype=int)
        cropping_array = np.asarray(cropping, dtype=int)

        if padding_array.any():
            padding_params = tuple(padding_array.tolist())
        else:
            padding_params = None

        if cropping_array.any():
            cropping_params = tuple(cropping_array.tolist())
        else:
            cropping_params = None

        # Apply padding if needed
        if padding_params is not None:
            subject = tio.Pad(padding_params)(subject)
        
        if cropping_params is not None:
            cropped_subject = tio.Crop(cropping_params)(subject)
        else:
            cropped_subject = subject

        if cropped_subject.spatial_shape != self.target_shape:
            # Resize to target shape
            resized_subject = tio.Resize(self.target_shape)(cropped_subject)
        else:
            resized_subject = cropped_subject

        return padding_params, resized_subject, subject_shape


    def apply_transform(self, data):
        raise NotImplementedError

class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            # import pdb; pdb.set_trace()
            # for dt in ["Tr", "Val", "Ts"]:
            for dt in ["Ts"]:
                d = os.path.join(path, f"labels{dt}")
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split(".nii.gz")[0]
                        label_path = os.path.join(path, f"labels{dt}", f"{base}.nii.gz")
                      
                        self.image_paths.append(label_path.replace("labels", "images"))
                        self.label_paths.append(label_path)

        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]


class Dataset_Union_ALL_Infer(Dataset):
    """Only for inference, no label is returned from __getitem__."""

    def __init__(
        self,
        paths,
        data_type="infer",
        image_size=128,
        transform=None,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])

        sitk_image_arr, _ = sitk_to_nib(sitk_image)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print("Could not transform", self.image_paths[index])

        # 计算实例的3D外接框
        label_data_np = subject.label.data.numpy()
        instances_not_filtered, instances_not_filtered_classes = seg_to_instances(label_data_np)
        final_mapping = {}
        if instances_not_filtered.max() > 0:
            boxes = get_bbox_np(instances_not_filtered[None])["boxes"]
        else:
            boxes = np.array([])
        
        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)
            
            return subject.image.data.clone().detach(), boxes
    
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "direction": sitk_image.GetDirection(),
                "origin": sitk_image.GetOrigin(),
                "spacing": sitk_image.GetSpacing(),
            }
            return subject.image.data.clone().detach(), boxes, meta_info
        else:
            return subject.image.data.clone().detach(), boxes, self.image_paths[index]

    def _set_file_paths(self, paths):
        self.image_paths = []

        # if ${path}/infer exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f"{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    image_path = os.path.join(
                        path, f"{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(image_path)
                    
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset):
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        # 计算实例的3D外接框
        label_data_np = subject.label.data.numpy()
        instances_not_filtered, instances_not_filtered_classes = seg_to_instances(label_data_np)
        final_mapping = {}
        if instances_not_filtered.max() > 0:
            boxes = get_bbox_np(instances_not_filtered[None])["boxes"]
        else:
            boxes = np.array([])
        
        
        return (
            subject.image.data.clone().detach(),
            subject.label.data.clone().detach(),
            boxes,
            self.image_paths[index],
        )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace("images", "labels"))


def img2label_paths(img_paths):
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
        seed=0,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(lambda i: (i, fcn(i)), self.indices)
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes * WORLD_SIZE
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Checks if available RAM is sufficient for caching images, adjusting for a safety margin."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files),
                bar_format=TQDM_BAR_FORMAT,
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """Fetches the dataset item at the given index, considering linear, shuffled, or weighted sampling."""
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.choice(self.indices)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        """Loads 1 image + 8 random images into a 9-image mosaic for augmented YOLOv5 training, returning labels and
        segments.
        """
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """Bundles a batch's data by quartering the number of shapes and paths, preparing it for model input."""
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                    0
                ].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / "coco128"):
    """Flattens a directory by copying all files from subdirectories to a new top-level directory, preserving
    filenames.
    """
    new_path = Path(f"{str(path)}_flat")
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f"{str(Path(path))}/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / "coco128"):
    """
    Converts a detection dataset to a classification dataset, creating a directory for each class and extracting
    bounding boxes.

    Example: from utils.dataloaders import *; extract_boxes()
    """
    path = Path(path)  # images dir
    shutil.rmtree(path / "classification") if (path / "classification").is_dir() else None  # remove existing
    files = list(path.rglob("*.*"))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / "classification") / f"{c}" / f"{path.stem}_{im_file.stem}_{j}.jpg"  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1] : b[3], b[0] : b[2]]), f"box failure in {f}"


def autosplit(path=DATASETS_DIR / "coco128/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


def verify_image_label(args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=['./data/medical_preprocessed/Dataset131_MICCAI_challenge_small/Tumor/MICCAI_challenge_small',],
        data_type='Tr',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(128,128,128)),
        ]),
        pcc=False,
        get_all_meta_info=True,
        split_idx = 0,
        split_num = 1,
        )

    # test_dataset = Dataset_Union_ALL_Val(
        # paths=["./data/validation/experimental/heart/hearts"],
        # mode="Val",
        # transform=tio.Compose(
            # [
                # tio.ToCanonical(),
                # tio.CropOrPad(target_shape=(128, 128, 128)),
            # ]
        # ),
        # threshold=0,
        # pcc=False,
        # get_all_meta_info=True,
    # )

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=4, shuffle=True
    )

    print(len(test_dataset))
    # import pdb; pdb.set_trace()
    
    # for i, j, n in test_dataloader:
    boxes_list = []
    for i, j, boxes in tqdm(test_dataloader):
        # print(i.shape)
        # print('boxes',boxes.shape)
        # print(j.shape)
        # print(n)
        # print(j.shape)
        boxes_list.append(boxes)
    import pdb; pdb.set_trace()