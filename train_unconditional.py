from rich import print

import jax
from jax import numpy as jnp

import flax
import flax.linen as nn
from flax.training import orbax_utils
import orbax.checkpoint

import einops

from typing import Literal

import numpy as np

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import datetime
from pathlib import Path

import tqdm
from math import sqrt

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax
from vqgan_jax.utils import custom_to_pil

from train_utils import build_train_step, create_train_state
from sample_utils import build_fast_sample_loop
from utils import dict_to_namespace, infinite_loader

import wandb

from absl import logging

# TODO: expand for whatever datasets we will use
# TODO: just pass config object
def get_data_loader(
    name: Literal["ffhq256"],
    batch_size: int = 1,
    num_workers: int = 0,
    train: bool = True,
):
    if name in ["ffhq256"]:
        dataset = ImageFolder(
            "data/ffhq256",
            transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
        )
        if train:
            dataset = Subset(dataset, list(range(60_000)))
        else:
            dataset = Subset(dataset, list(range(60_000, 70_000)))
    else:
        raise ValueError(f"unrecognised dataset name '{name}'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        shuffle=True,
    )
    return dataset, loader


def main(config):
    # jax.distributed.initialize()

    devices = jax.devices()
    replication_factor = len(devices)

    key = jax.random.PRNGKey(config.seed)

    if config.enable_checkpointing:
        # TODO: add drive root param
        save_name = Path("/mnt/disks/persist/checkpoints") / datetime.datetime.now().strftime("sundae-checkpoints_%Y-%d-%m_%H-%M-%S")
        save_name.mkdir()
        logging.info(f"Working directory '{save_name}'")
        orbax_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        checkpoint_opts = orbax.checkpoint.CheckpointManagerOptions(
            keep_period=config.checkpoint.keep_period,
            max_to_keep=config.checkpoint.max_to_keep,
            create=True,
        )
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            save_name, orbax_checkpointer, checkpoint_opts
        )

    logging.info(f"Loading dataset '{config.data.name}'")
    _, train_loader = get_data_loader(
        config.data.name, config.batch_size, config.data.num_workers, train=True
    )
    _, eval_loader = get_data_loader(
        config.data.name,
        config.batch_size * 2,
        config.data.num_workers,
        train=False,
    )

    train_iter = infinite_loader(train_loader)
    eval_iter = infinite_loader(eval_loader)

    logging.info(f"Loading VQ-GAN")
    vqgan_dtype = getattr(jnp, config.vqgan.dtype)
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
        config.vqgan.name, dtype=vqgan_dtype
    )

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey, config)
    if config.enable_checkpointing:
        save_args = orbax_utils.save_args_from_target(state)

    logging.info(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")

    train_step = build_train_step(config, vqgan=vqgan, train=True)
    eval_step = build_train_step(config, vqgan=vqgan, train=False)
    # TODO: param all this
    sample_loop = build_fast_sample_loop(config, vqgan=vqgan, temperature=0.7, proportion=0.5)
    state = flax.jax_utils.replicate(state)

    if config.report_to_wandb:
        wandb.init(project="diffusers-sprint-sundae", config=config)
    else:
        wandb.init(mode="disabled")

    pmap_train_step = jax.pmap(train_step, "replication_axis", in_axes=(0, 0, 0))
    pmap_eval_step = jax.pmap(eval_step, "replication_axis", in_axes=(0, 0, 0))
    pmap_sample_loop = jax.pmap(sample_loop, "replication_axis", in_axes=(0, 0))

    step = 0
    while step < config.training.steps:
        metrics = dict(loss=0.0, accuracy=0.0)
        pb = tqdm.trange(config.training.batches[0])
        for i in pb:
            batch, _ = next(train_iter)
            batch = einops.rearrange(
                batch.numpy(), "(r b) c h w -> r b c h w", r=replication_factor, c=3
            )
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            state, loss, accuracy = pmap_train_step(
                state, batch, subkeys
            )  # TODO: add donate args, memory save on params
            loss, accuracy = loss.mean(), accuracy.mean()

            metrics["loss"] += loss
            metrics["accuracy"] += accuracy

            pb.set_description(
                f"[step {step+1:,}/{config.training.steps:,}] [train] loss: {metrics['loss'] / (i+1):.6f}, accuracy {metrics['accuracy'] / (i+1):.2f}"
            )
            step += 1

        wandb.log(
            {
                "train": {
                    "loss": metrics["loss"] / (i + 1),
                    "accuracy": metrics["accuracy"] / (i + 1),
                }
            },
            commit=False,
            step=step,
        )

        if config.enable_checkpointing:
            checkpoint_manager.save(
                step,
                flax.jax_utils.unreplicate(state),
                save_kwargs={"save_args": save_args},
            )

        metrics = dict(loss=0.0, accuracy=0.0)
        pb = tqdm.trange(config.training.batches[1])
        for i in pb:
            batch, _ = next(eval_iter)
            batch = einops.rearrange(
                batch.numpy(), "(r b) c h w -> r b c h w", r=replication_factor, c=3
            )
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            loss, accuracy = pmap_eval_step(state, batch, subkeys)

            loss, accuracy = loss.mean(), accuracy.mean()

            metrics["loss"] += loss
            metrics["accuracy"] += accuracy

            pb.set_description(
                f"[step {step+1:,}/{config.training.steps:,}] [eval] loss: {metrics['loss'] / (i+1):.6f}, accuracy {metrics['accuracy'] / (i+1):.2f}"
            )

        wandb.log(
            {
                "eval": {
                    "loss": metrics["loss"] / (i + 1),
                    "accuracy": metrics["accuracy"] / (i + 1),
                }
            },
            commit=False,
            step=step,
        )

        logging.info("sampling from current model")
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, replication_factor)
        img = pmap_sample_loop(state.params, subkeys)
        img = jnp.reshape(img, (-1, config.data.image_size, config.data.image_size, 3))
        sqrt_num_images = int(sqrt(img.shape[0]))
        img = custom_to_pil(
            np.asarray(
                einops.rearrange(
                    img, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=sqrt_num_images, b2=img.shape[0] // sqrt_num_images
                )
            )
        )
        if config.enable_checkpointing:
            img.save(Path(save_name) / f"sample-{step:08}.jpg")
        wandb.log({"sample": wandb.Image(img)}, commit=True, step=step)


if __name__ == "__main__":
    # TODO: add proper argparsing!:
    # TODO: this sadly breaks some hierarchical config arguments :/ really we
    # need something like a yaml config loader or whatever format. Or use
    # SimpleParsing and we can have config files with arg overrides which are
    # also hierarchical

    config = dict(
        data=dict(
            name="ffhq256",
            batch_size=32,  # TODO: really this shouldn't be under data, it affects the numerics of the model
            num_workers=4,
        ),
        model=dict(
            num_tokens=16384,
            dim=1024,
            depth=[2, 10, 2],
            shorten_factor=2,
            resample_type="linear",
            heads=8,
            dim_head=64,
            rotary_emb_dim=32,
            max_seq_len=16,  # effectively squared to 256
            parallel_block=False,
            tied_embedding=False,
            dtype=jnp.bfloat16,  # currently no effect
        ),
        training=dict(
            learning_rate=4e-4,
            end_learning_rate=3e-6,
            warmup_start_lr=1e-6,
            warmup_steps=4000,
            unroll_steps=3,
            steps=1_000_000,
            max_grad_norm=5.0,
            weight_decay=0.0,
            temperature=1.0,
            batches=(4000, 200),
        ),
        checkpoint=dict(keep_period=100, max_to_keep=3),
        vqgan=dict(name="vq-f16", dtype=jnp.bfloat16),
        report_to_wandb=True,
        seed=42
    )

    main(dict_to_namespace(config))
