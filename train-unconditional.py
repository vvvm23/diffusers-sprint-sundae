import jax
from jax import lax, numpy as jnp
from jax.typing import ArrayLike

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints

import optax

from typing import Callable, Optional, Sequence, Union, Literal

import numpy as np

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, ImageFolder

import datetime
from pathlib import Path

import tqdm

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax

from sundae import SundaeModel
from train_utils import build_train_step
from utils import dict_to_namespace

# TODO: expand for whatever datasets we will use
# TODO: auto train-valid split
def get_data_loader(name: Literal['ffhq256'], batch_size: int = 1, num_workers: int = 0):
    if name in ['ffhq256']:
        dataset = ImageFolder('data/ffhq256', transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]))
    else:
        raise ValueError(f"unrecognised dataset name '{name}'")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return dataset, loader

def create_train_state(key, config: dict):
    model = SundaeModel(config.model)
    params = model.init(key, jnp.zeros([1, config.model.max_seq_len*config.model.max_seq_len], dtype=jnp.int32))['params']
    params = jax.tree_map(lambda p: jnp.asarray(p, dtype=config.model.dtype), params)
    opt = optax.adamw(config.training.learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


def main(config, args):
    print("Config:", config)

    print('JAX devices:', jax.devices())
    
    key = jax.random.PRNGKey(args.seed)
    print('Random seed:', args.seed)

    print(f"Loading dataset '{config.data.name}'")
    _, loader = get_data_loader(config.data.name, config.data.batch_size, config.data.num_workers)

    print(f"Loading VQ-GAN")
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(config.vqgan.name, dtype=config.vqgan.dtype)

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey, config)

    save_name = datetime.datetime.now().strftime("sundae-checkpoints_%Y-%d-%m_%H-%M-%S")
    Path(save_name).mkdir()
    print(f"Saving checkpoints to directory {save_name}")

    train_step = build_train_step(config, vqgan)

    # TODO: wandb logging plz
    for ei in range(config.training.epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        pb = tqdm.tqdm(loader)
        for i, (batch, _) in enumerate(pb):
            batch = batch.numpy()
            key, subkey = jax.random.split(key)
            state, loss, accuracy = train_step(state, batch, subkey) 
            total_loss += loss
            total_accuracy += accuracy
            pb.set_description(f"[epoch {ei+1}] loss: {total_loss / (i+1):.6f}, accuracy {total_accuracy / (i+1):.2f}")

        checkpoints.save_checkpoint(ckpt_dir=save_name, target=state, step = ei)

if __name__ == '__main__':
    # TODO: add proper argparsing!
    config = dict(
        data = dict(
            name = 'ffhq256',
            batch_size = 48, # TODO: really this shouldn't be under data
            num_workers = 4
        ),
        model = dict(
            num_tokens=16_384, 
            dim=1024,
            depth=[2, 10, 2], 
            shorten_factor = 4, 
            resample_type = 'linear',
            heads = 8,
            dim_head = 64,
            rotary_emb_dim = 32,
            max_seq_len = 16,
            parallel_block = True,
            tied_embedding = False,
            dtype = jnp.bfloat16,
        ),
        training = dict(
            learning_rate = 4e-4,
            unroll_steps = 2,
            epochs = 100 # TODO: maybe replace with train steps
        ),
        vqgan = dict(
            name = 'vq-f16',
            dtype = jnp.bfloat16
        ),
        jit_enabled = True
    )

    args = dict(
        seed = 0xffff
    )

    config, args = dict_to_namespace(config), dict_to_namespace(args)

    main(config, args)
