# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pytorchvideo
import torch
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pytorchvideo.data import ClipSampler
from pytorchvideo.data import LabeledVideoDataset, labeled_video_dataset
from typing import Any, Callable, Dict, Optional, Type

from pytorchvideo.transforms import (
    ShortSideScale,
    UniformCropVideo,
    Normalize,
    ApplyTransformToKey,
)

from torchvision.transforms import (
    Compose,
    Lambda,
)


def RAVDESS(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for the RAVDESS dataset.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.RAVDESS")

    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    Lambda(lambda x: x[[0, len(x)//2,-1]]),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=256),
                    UniformCropVideo(size=256, aug_index_key=1) # aug_index_key=1 for center
                  ]
                ),
              ),
            ]
        )

def create_csv(data_path):
    import os
    import glob
    import csv

    csv_filename = 'train.csv'
    if os.path.exists(csv_filename):
        return
    
    with open(csv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for filename in glob.glob(os.path.join(data_path, '*', '*.mp4')):
            writer.writerow([filename, 0])

data_path = './data'
create_csv(data_path)
delta = 500e-3  # 500ms

video_train_dataset = RAVDESS(
            data_path=os.path.join(data_path, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", 2*delta),
            transform=train_transform
        )
