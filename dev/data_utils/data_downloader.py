import sys
import os
from datetime import datetime
import pandas as pd
import contexttimer
from urllib.request import urlopen
import requests
from PIL import Image
# import torch
# from torchvision.transforms import functional as TF
from multiprocessing import Pool
from tqdm import tqdm
import logging
import sys

from datasets import load_dataset

# Setup
logging.basicConfig(filename='download.log', filemode='w', level=logging.INFO)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


# Load data
print(f'Starting to load at {datetime.now().isoformat(timespec="minutes")}')

with contexttimer.Timer(prefix="Loading from HF dataset:"):
    df = load_dataset('laion/laion-art', split='train').to_pandas()
    df = df[["URL", "TEXT"]]

url_to_idx_map = {url: index for index, url, caption in df.itertuples()}
print(f'Loaded {len(url_to_idx_map)} urls')

# create base directory if it doesn't exist
base_dir = os.path.join(os.getcwd(), "laion-art-images")
if not os.path.isdir(base_dir):
    os.makedirs(base_dir)

def process(item):
    url, image_id = item
    try:
        base_url = os.path.basename(url)  # extract base url
        stem, ext = os.path.splitext(base_url)  # split into stem and extension
        filename = f'{image_id:08d}---{stem}.jpg'  # create filename
        filepath = os.path.join(base_dir, filename)  # concat to get filepath
        if not os.path.isfile(filepath):
            req = requests.get(url, stream=True, timeout=1, verify=False).raw
            image = Image.open(req).convert('RGB')
            if min(image.size) > 512:
                image = TF.resize(image, size=512, interpolation=Image.LANCZOS)
            image.save(filepath)  # save PIL image
    except Exception as e:
        logging.info(" ".join(repr(e).splitlines()))
        logging.error(url)

list_of_items = list(url_to_idx_map.items())
print("a sample from the download list:")
print(list_of_items[0])
print(len(list_of_items))

with Pool(128) as p:
    r = list(tqdm(p.imap(process, list_of_items), total=len(list_of_items)))
    print('DONE')