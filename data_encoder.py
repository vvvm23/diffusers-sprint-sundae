import io
from pathlib import Path
import requests
from PIL import Image, ImageFile
from PIL.Image import DecompressionBombWarning
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import os

import jax
import jax.numpy as jnp
from jax import pmap

import warnings
from typing import Optional, Callable

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DecompressionBombWarning)


import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax

print("Loading VQGAN model...")
vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
    "vq-f16", dtype=jnp.float32
)

laion_art_images = Path("/mnt/disks/persist/laion-art-images")
laion_art_output = Path("/mnt/disks/persist/laion_art_images_vqganf16_encoded")


class LaionArtDataset(Dataset):
    """
    A PyTorch Dataset class for (image, texts) tasks. Note that this dataset
    returns the raw text rather than tokens. This is done on purpose, because
    it's easy to tokenize a batch of text after loading it from this dataset.
    """

    def __init__(
        self,
        *,
        images_root: str,
        captions_path: str,
        text_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
        image_transform_type: str = "torchvision",
        include_captions: bool = True,
    ):
        """
        :param images_root: folder where images are stored
        :param captions_path: path to csv that maps image filenames to captions
        :param image_transform: image transform pipeline
        :param text_transform: image transform pipeline
        :param image_transform_type: image transform type, either `torchvision` or `albumentations`
        :param include_captions: Returns a dictionary with `image`, `text` if `true`; otherwise returns just the images.
        """

        # Base path for images
        self.images_root = Path(images_root)

        # Load captions as DataFrame
        self.captions = pd.read_csv(captions_path, sep="\t")
        self.captions["image_file"] = self.captions["image_file"].astype(str)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.image_transform_type = image_transform_type.lower()
        assert self.image_transform_type in ["torchvision", "albumentations"]

        # Total number of datapoints
        self.size = len(self.captions)

        # Return image+captions or just images
        self.include_captions = include_captions

    def image_exists(self, item):
        url, caption, image_id = item
        base_url = os.path.basename(url)  # extract base url
        stem, ext = os.path.splitext(base_url)  # split into stem and extension
        filename = f"{image_id:08d}---{stem}.jpg"  # create filename
        base_dir = str(self.images_root)
        image_path = Path(base_dir, filename)  # concat to get filepath

        return image_path.exists()

    def verify_that_all_images_exist(self):
        for image_file in self.captions["image_file"]:
            if not image_exists:
                print(f"file does not exist: {image_file}")

    def _get_raw_image(self, i):
        base_url, image_id = (
            os.path.basename(self.captions.iloc[i]["image_file"]),
            self.captions.iloc[i]["image_id"],
        )
        stem, ext = os.path.splitext(base_url)
        image_path = Path(self.images_root, f"{image_id:08d}---{stem}.jpg")
        image = default_loader(image_path)
        return image

    def _get_raw_text(self, i):
        return self.captions.iloc[i]["caption"]

    def __getitem__(self, i):
        image = self._get_raw_image(i)
        caption = self._get_raw_text(i)
        image_id = self.captions.iloc[i]["image_id"]
        if self.image_transform is not None:
            if self.image_transform_type == "torchvision":
                image = self.image_transform(image)
            elif self.image_transform_type == "albumentations":
                image = self.image_transform(image=np.array(image))["image"]
            else:
                raise NotImplementedError(f"{self.image_transform_type=}")
        return (
            {"image": image, "text": caption, "image_id": image_id}
            if self.include_captions
            else image
        )

    def __len__(self):
        return self.size


images_root = "/mnt/disks/persist/laion-art-images/"
captions_path = "./laion_art_clean.tsv"
image_size = 256


# Create transforms
def image_transform(image):
    s = min(image.size)
    r = image_size / s
    s = (round(r * image.size[1]), round(r * image.size[0]))
    image = TF.resize(image, s, interpolation=InterpolationMode.LANCZOS)
    image = TF.center_crop(image, output_size=2 * [image_size])
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    image = image.permute(0, 2, 3, 1).numpy()
    return image


# Create dataset
print("Creating dataset")
dataset = LaionArtDataset(
    images_root=images_root,
    captions_path=captions_path,
    image_transform=image_transform,
    image_transform_type="torchvision",
    include_captions=False,
)


def encode(model, batch):
    _, indices = model.encode(batch)
    return indices


def superbatch_generator(dataloader, num_tpus):
    iter_loader = iter(dataloader)
    for batch in iter_loader:
        superbatch = [batch.squeeze(1)]
        try:
            for b in range(num_tpus - 1):
                batch = next(iter_loader)
                if batch is None:
                    break
                # Skip incomplete last batch
                if batch.shape[0] == dataloader.batch_size:
                    superbatch.append(batch.squeeze(1))
        except StopIteration:
            pass
        superbatch = torch.stack(superbatch, axis=0)
        yield superbatch


import os
import time


def encode_captioned_dataset(
    dataset, output_tsv, batch_size=32, num_workers=16, save_frequency=1000
):
    if os.path.isfile(output_tsv):
        print(f"Destination file {output_tsv} already exists, please move away.")
        return

    num_tpus = jax.device_count()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    superbatches = superbatch_generator(dataloader, num_tpus=num_tpus)
    p_encoder = pmap(lambda batch: encode(vqgan, batch))

    all_captions = []
    all_image_ids = []
    all_encoding = []
    n_file = 1
    # We save each superbatch to avoid reallocation of buffers as we process them.
    # We keep the file open to prevent excessive file seeks.

    iterations = len(dataset) // (batch_size * num_tpus)
    for n in tqdm(range(iterations)):
        superbatch = next(superbatches)
        encoded = p_encoder(superbatch.numpy())
        encoded = encoded.reshape(-1, encoded.shape[-1])

        # Extract fields from the dataset internal `captions` property, and save to disk
        start_index = n * batch_size * num_tpus
        end_index = (n + 1) * batch_size * num_tpus
        paths = dataset.captions["image_file"][start_index:end_index].values
        captions = dataset.captions["caption"][start_index:end_index].values
        image_ids = dataset.captions["image_id"][start_index:end_index].values

        all_captions.extend(list(captions))
        all_encoding.extend(encoded.tolist())
        all_image_ids.extend(list(image_ids))

        # save files
        if (n + 1) % save_frequency == 0:
            # from IPython import embed; embed()
            print(f"Saving file {n_file}")
            batch_df = pd.DataFrame.from_dict(
                {
                    "caption": all_captions,
                    "encoding": all_encoding,
                    "image_id": all_image_ids,
                }
            )
            batch_df.to_parquet(f"{laion_art_output}/{n_file:03d}.parquet")
            all_captions = []
            all_encoding = []
            all_image_ids = []
            n_file += 1

    if len(all_captions):
        print(f"Saving final file {n_file}")
        batch_df = pd.DataFrame.from_dict(
            {
                "caption": all_captions,
                "encoding": all_encoding,
                "image_id": all_image_ids,
            }
        )
        batch_df.to_parquet(f"{laion_art_output}/{n_file:03d}.parquet")


print(f"device_count: {jax.device_count()}")
print("Encoding dataset...")
batch_size = 128  # Per device
num_workers = 16  # For parallel processing
save_frequency = 128  # Number of batches to create a new file
encode_captioned_dataset(
    dataset,
    laion_art_output,
    batch_size=batch_size,
    num_workers=num_workers,
    save_frequency=save_frequency,
)
