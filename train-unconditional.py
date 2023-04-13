from typing import (
    Callable, 
    Optional, 
    Sequence, 
    Union, 
    Literal
)

import datetime
from pathlib import Path

import numpy as np

import jax
from jax import (
    lax, 
    numpy as jnp
)
from jax.typing import ArrayLike

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints

import optax

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import (
    MNIST, 
    ImageFolder
)

import tqdm

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax

from sundae import SundaeModel
from train_utils import build_train_step
from utils import dict_to_namespace

# Hatman: added imports for wandb, argparse, and bunch
import wandb
import argparse
from bunch import Bunch


# TODO: expand for whatever datasets we will use
# TODO: auto train-valid split
def get_data_loader(
    name: Literal["ffhq256"], batch_size: int = 1, num_workers: int = 0
):
    if name in ["ffhq256"]:
        dataset = ImageFolder(
            "data/ffhq256",
            transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
        )
    else:
        raise ValueError(f"unrecognised dataset name '{name}'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return dataset, loader


def create_train_state(key, config: dict):
    model = SundaeModel(config.model)
    params = model.init(
        key,
        jnp.zeros(
            [1, config.model.max_seq_len * config.model.max_seq_len], dtype=jnp.int32
        ),
    )["params"]
    params = jax.tree_map(lambda p: jnp.asarray(p, dtype=config.model.dtype), params)
    opt = optax.adamw(config.training.learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


def main(config, args):
    print("Config:", config)
    print("Args:", args)

    print("JAX devices:", jax.devices())

    key = jax.random.PRNGKey(args.seed)
    print("Random seed:", args.seed)

    print(f"Loading dataset '{config.data.name}'")
    _, loader = get_data_loader(
        config.data.name, config.data.batch_size, config.data.num_workers
    )

    print(f"Loading VQ-GAN")
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
        config.vqgan.name, dtype=config.vqgan.dtype
    )

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey, config)

    save_name = datetime.datetime.now().strftime("sundae-checkpoints_%Y-%d-%m_%H-%M-%S")
    Path(save_name).mkdir()
    print(f"Saving checkpoints to directory {save_name}")
    train_step = build_train_step(config, vqgan)

    # TODO: wandb logging plz: Hatman
    # TODO: need flag to toggle on and off otherwise we will pollute project
    # wandb.init(project="diffusers-sprint-sundae", config=config)

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
            pb.set_description(
                f"[epoch {ei+1}] loss: {total_loss / (i+1):.6f}, accuracy {total_accuracy / (i+1):.2f}"
            )
            # TODO: wandb logging plz: Hatman
            # TODO: we should log the raw value of loss/acc for wandb, not scaled by (i+1)
            # TODO: or really, we should log every N and avg over that
            # wandb.log({"loss": total_loss / (i+1), "accuracy": total_accuracy / (i+1)})

        checkpoints.save_checkpoint(ckpt_dir=save_name, target=state, step=ei, keep=5) # TODO: param this


if __name__ == "__main__":
    # TODO: add proper argparsing!: Hatman
    # TODO: this sadly breaks some hierarchical config arguments :/ really we
    # need something like a yaml config loader or whatever format. Or use
    # SimpleParsing and we can have config files with arg overrides which are
    # also hierarchical

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-n", "--name", type=str, default="ffhq256")
    parser.add_argument("-b", "--batch_size", type=int, default=48)
    parser.add_argument("-w", "--num_workers", type=int, default=4)
    parser.add_argument("-t", "--num_tokens", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("-d", "--depth", type=int, default=[2, 10, 2])
    parser.add_argument("-sf", "--shorten_factor", type=int, default=4)
    parser.add_argument("-rt", "--resample_type", type=str, default="linear")
    parser.add_argument("-h", "--heads", type=int, default=8)
    parser.add_argument("-dh", "--dim_head", type=int, default=64)
    parser.add_argument("-red", "--rotary_emb_dim", type=int, default=32)
    parser.add_argument("-msl", "--max_seq_len", type=int, default=16)
    parser.add_argument("-pb", "--parallel_block", type=bool, default=True)
    parser.add_argument("-te", "--tied_embedding", type=bool, default=False)
    parser.add_argument("-dt", "--dtype", type=str, default="bfloat16")
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-4)
    parser.add_argument("-us", "--unroll_steps", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-vq", "--vqgan_name", type=str, default="vq-f16")
    parser.add_argument("-vqdt", "--vqgan_dtype", type=str, default="bfloat16")
    parser.add_argument("-jit", "--jit_enabled", type=bool, default=True)
    config = parser.parse_args()
    """
    config = dict(
        data=dict(
            name="ffhq256",
            batch_size=48,  # TODO: really this shouldn't be under data, it affects the numerics of the model
            num_workers=4,
        ),
        model=dict(
            num_tokens=16_384,
            dim=1024,
            depth=[2, 12, 2],
            shorten_factor=4,
            resample_type="linear",
            heads=8,
            dim_head=64,
            rotary_emb_dim=32,
            max_seq_len=16, # effectively squared to 256
            parallel_block=True,
            tied_embedding=False,
            dtype=jnp.bfloat16,
        ),
        training=dict(
            learning_rate=1e-4,
            unroll_steps=2,
            epochs=100,  # TODO: maybe replace with train steps
        ),
        vqgan=dict(name="vq-f16", dtype=jnp.bfloat16),
        jit_enabled=True,
    )

    # Hatman: To eliminate dict_to_namespace
    args = Bunch(dict(seed=0xFFFF))
    # args = dict_to_namespace()

    main(dict_to_namespace(config), args)
