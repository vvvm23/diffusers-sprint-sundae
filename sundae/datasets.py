from typing import (
    Any,
    List,
    Dict,
    Union,
    Optional,
    Callable
)

import os
import math
import urllib
import logging
import requests

import numpy as np

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict
)

import ml_collections as mlc


logger = logging.getLogger(__name__)


ImageTransform = Callable[[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor]


def is_url(path: str) -> bool:
    return urllib.parse.urlparse(url).scheme != ""

def read_image(path: str) -> Image.Image:
    if is_url(image_path):
        image_data = requests.get(image_path, stream=True).raw
        image = Image.open(image_data).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

class ImageTransform(torch.nn.Module):
    def __init__(self, transforms: List[torch.nn.Module]) -> None:
        super().__init__()
        self.transforms = torch.nn.Sequential(*transforms)

    def forward(self, x) -> torch.Tensor:
        '''Transform x of type `PIL.Image.Image`.'''
        with torch.no_grad():
            x = self.transforms(x)
            x = x / 127.5 - 1.0
        return x


def training_image_transforms(config: mlc.ConfigDict) -> List[torch.nn.Module]:
    return [
        transforms.RandomResizedCrop(config.data.image_size, antialias=True),
        transforms.RandomHorizontalFlip(config.data.flip_p),
    ]


def validation_image_transforms(config: mlc.ConfigDict) -> List[torch.nn.Module]:
    return [
        transforms.Resize([config.data.image_size], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.data.image_size),
    ]


def jitted_image_transform(config: mlc.ConfigDict, split: Optional[str] = None) -> torch.nn.Module:
    if split == "train":
        transforms = training_image_transforms(config)
    else:
        transforms = validation_image_transforms(config)    
    image_transform = ImageTransform(transforms)
    image_transform = torch.jit.script(image_transform)
    return image_transform


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
    splits: Optional[List[str]] = None,
    image_column: str = "image",
    overwrite_cache: bool = False
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
    if splits is None:
        splits = list(datasets.keys())

    preprocessed_datasets = datasets.__class__()
    for split, dataset in datasets.items():
        if split not in splits:
            continue
        
        max_samples_split = max_samples[split]
        transform_images_split = transform_images[split]
        
        if max_samples_split is not None:
            max_samples_split = min(hf_dataset_length(dataset), max_samples_split)
            dataset = dataset.select(range(max_samples_split))
        if transform_images_split is not None:
            if isinstance(dataset, IterableDataset):
                def preprocessing_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
                    images = []
                    discarded = []
                    for image in examples[image_column]:
                        try:
                            if isinstance(image, str):
                                image = read_image(image)
                            image = TF.to_tensor(image)
                            image = transform_images_split(image)
                            images.append(image)
                            discarded.append(False)
                        except Exception as e:
                            logger.info(f'Caught: "{e}" during image reading.')
                            discarded.append(True)
                    
                    # Discard samples were couldn't read the image
                    # for name, values in examples.items():
                    #     if name != image_column:
                    #         tmp = []
                    #         for discard, value in zip(discarded, values):
                    #             if not discard:
                    #                 tmp.append(value)
                    #         examples[name] = tmp

                    examples["pixel_values"] = images
                    return examples

                dataset = dataset.remove_columns([c for c in dataset.column_names if c != image_column])
                dataset = dataset.map(
                    preprocessing_fn,
                    batched=True,
                    remove_columns=[image_column]
                )
            else:
                # The dataset is of type `Dataset`
                dataset = dataset.remove_columns([c for c in dataset.column_names if c != image_column])
                dataset.set_transform(transform_images_split)
            
        preprocessed_datasets[split] = dataset
    return preprocessed_datasets
