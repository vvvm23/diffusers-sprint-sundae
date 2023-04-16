from typing import (
    List,
    Dict,
    Union,
    Optional,
    Callable
)

import math

import numpy as np

import torch
import torchvision
from torchvision import transforms

from PIL import Image

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict
)

import ml_collections as mlc


ImageTransform = Callable[[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor]


class ImageTransform(torch.nn.Module):
    def __init__(self, transforms: List[torch.nn.Module]) -> None:
        super().__init__()
        self.transforms = torch.nn.Sequential(*transforms)

    def forward(self, x: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            x = x / 127.5 - 1.0
        return x


def train_image_transforms(config: mlc.ConfigDict) -> List[torch.nn.Module]:
    return [
        transforms.RandomResizedCrop(config.data.image_size, antialias=True),
        transforms.RandomHorizontalFlip(config.data.flip_p),
    ]


def validation_image_transforms(config: mlc.ConfigDict) -> List[torch.nn.Module]:
    return [
        transforms.Resize([config.data.image_size], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.data.image_size),
    ]


def hf_dataset_length(dataset: Union[Dataset, IterableDataset]) -> Union[float, int]:
    try:
        return len(dataset)
    except TypeError as e:
        logger.info(f'Caught: "{e}". Returning `math.inf` instead!')
        return math.inf


def load_hf_unconditional(
    dataset_name: str,
    dataset_config_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
    keep_in_memory: bool = False,
    use_auth_token: Optional[bool] = None,
    streaming: bool = False,
    num_processes: Optional[int] = None,
    max_samples: Optional[Union[int, Dict[str, int]]] = None,
    transform_images: Union[ImageTransform, Dict[str, ImageTransform]] = None,
    splits: Optional[List[str]] = None
) -> Union[DatasetDict, IterableDatasetDict]:
    # Load the dataset dict
    datasets = load_dataset(
        path=dataset_name,
        name=dataset_config_name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        use_auth_token=use_auth_token,
        streaming=streaming,
        num_proc=num_processes
    )

    # Format args as dictionary if required
    if not isinstance(max_samples, dict):
        max_samples = {key: max_samples for key in datasets.keys()}
    if not isinstance(transform_images, dict):
        transform_images = {key: transform_images for key in datasets.keys()}
    if split is None:
        splits = list(datasets.keys())

    preprocessed_datasets = datasets.__class__()
    for split, dataset in enumerate(datasets):
        if split not in splits:
            continue
        
        max_samples_split = max_samples[split]
        transform_images_split = transform_images[split]
        
        if max_samples_split is not None:
            max_samples_split = min(hf_dataset_length(dataset), max_samples_split)
            dataset = dataset.select(range(max_samples_split))
        if transform_images_split is not None:
            dataset.set_transform(transform_images_split)

        preprocessed_datasets[split] = dataset
    return preprocessed_datasets
