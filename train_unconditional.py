from rich import print

import jax
from jax import make_jaxpr
from jax import numpy as jnp

import flax
import flax.linen as nn
from flax.training import orbax_utils
import orbax.checkpoint

import optax
import einops

from typing import Callable, Optional, Sequence, Union, Literal

import numpy as np

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import datetime
from pathlib import Path

import tqdm

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax
from vqgan_jax.utils import custom_to_pil

from train_utils import build_train_step, create_train_state
from utils import dict_to_namespace, infinite_loader
from sundae.model import SundaeModel

import wandb
from bunch import Bunch


# TODO: expand for whatever datasets we will use
def get_data_loader(
    name: Literal["ffhq256"],
    batch_size: int = 1,
    num_workers: int = 0,
    train: bool = True,
):
    if name in ["ffhq256"]:
        dataset = ImageFolder(
            "data/ffhq256",
            # transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
            transform=T.Compose([T.ToTensor()]),
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


def main(config, args):
    print("Config:", config)
    print("Args:", args)

    devices = jax.devices()
    print("JAX devices:", devices)
    replication_factor = len(devices)

    key = jax.random.PRNGKey(args.seed)
    print("Random seed:", args.seed)

    print(f"Loading dataset '{config.data.name}'")
    _, train_loader = get_data_loader(
        config.data.name, config.data.batch_size, config.data.num_workers, train=True
    )
    _, eval_loader = get_data_loader(
        config.data.name,
        config.data.batch_size * 2,
        config.data.num_workers,
        train=False,
    )

    train_iter = infinite_loader(train_loader)
    eval_iter = infinite_loader(eval_loader)

    print(f"Loading VQ-GAN")
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
        config.vqgan.name, dtype=config.vqgan.dtype
    )

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey, config)

    print(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")

    save_name = datetime.datetime.now().strftime("sundae-checkpoints_%Y-%d-%m_%H-%M-%S")
    Path(save_name).mkdir()
    print(f"Saving checkpoints to directory {save_name}")
    train_step = build_train_step(config, vqgan=vqgan, train=True)
    eval_step = build_train_step(config, vqgan=vqgan, train=False)
    state = flax.jax_utils.replicate(state)

    if args.wandb:
        wandb.init(project="diffusers-sprint-sundae", config=config)
    else:
        wandb.init(mode="disabled")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_opts = orbax.checkpoint.CheckpointManagerOptions(
        keep_period=config.checkpoint.keep_period,
        max_to_keep=config.checkpoint.max_to_keep,
        create=True,
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        save_name, orbax_checkpointer, checkpoint_opts
    )
    save_args = orbax_utils.save_args_from_target(state)

    pmap_train_step = jax.pmap(train_step, "replication_axis", in_axes=(0, 0, 0))
    pmap_eval_step = jax.pmap(eval_step, "replication_axis", in_axes=(0, 0, 0))

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

        # TODO: pjit sample?
        print("sampling from current model")
        key, subkey = jax.random.split(key)
        # TODO: param all this
        sample_model = SundaeModel(config.model)
        sample_model.params = flax.jax_utils.unreplicate(
            state
        ).params  # TODO: do we need to unreplicate all? or just params?
        sample = sample_model.sample(
            subkey,
            num_samples=4,
            steps=100,
            temperature=0.7,
            proportion=0.4,
            early_stop=False,
            progress=False,
            return_history=False,
        )  # TODO: param this
        decoded_image = jax.jit(vqgan.decode_code)(sample)
        img = custom_to_pil(
            np.asarray(
                einops.rearrange(
                    decoded_image, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=2, b2=2
                )
            )
        )
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
            batch_size=16,  # TODO: really this shouldn't be under data, it affects the numerics of the model
            num_workers=4,
        ),
        model=dict(
            num_tokens=16384,
            dim=1024,
            depth=[2, 10, 2],
            shorten_factor=4,
            resample_type="linear",
            heads=8,
            dim_head=128,
            rotary_emb_dim=64,
            max_seq_len=32,  # effectively squared to 256
            parallel_block=False,
            tied_embedding=False,
            dtype=jnp.bfloat16,  # currently no effect
        ),
        training=dict(
            learning_rate=4e-4,
            end_learning_rate=3e-6,
            warmup_start_lr=1e-6,
            warmup_steps=5000,
            unroll_steps=2,
            steps=1_000_000,
            max_grad_norm=5.0,
            weight_decay=1e-2,
            temperature=0.5,
            batches=(1000, 50),
        ),
        checkpoint=dict(keep_period=50, max_to_keep=3),
        vqgan=dict(name="vq-f8", dtype=jnp.bfloat16),
    )

    args = Bunch(
        dict(wandb=True, seed=42)
    )  # if you are changing the seed to get good results, may god help you, hallelujah.

    main(dict_to_namespace(config), args)
