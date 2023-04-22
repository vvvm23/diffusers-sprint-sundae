import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

import flax
import flax.linen as nn
from flax.training import orbax_utils
import orbax.checkpoint

import einops

from typing import Callable, Optional, Sequence, Union, Literal
from absl import logging

import numpy as np

from transformers import (
    FlaxCLIPTextModel,
    CLIPTokenizer,
    CLIPTokenizerFast
)

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import datetime
from pathlib import Path
from math import sqrt

import tqdm

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax
from vqgan_jax.utils import custom_to_pil

from train_utils import build_train_step, create_train_state
from sample_utils import build_fast_sample_loop
from utils import dict_to_namespace, infinite_loader

# Hatman: added imports for wandb, argparse, and bunch
import wandb
from bunch import Bunch

def load_text_encoder(config) -> FlaxCLIPTextModel:
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.text_encoder.model_name_or_path, 
        from_pt=config.text_encoder.from_pt
    )
    return text_encoder

def load_tokenizer(config) -> Union[CLIPTokenizer, CLIPTokenizerFast]:
    Tokenizer = CLIPTokenizerFast if config.text_encoder.use_fast_tokenizer else CLIPTokenizer
    tokenizer = Tokenizer.from_pretrained(config.text_encoder.model_name_or_path)
    return tokenizer

def compute_classifer_free_embedding(config, encoder: FlaxCLIPTextModel, tokenizer: CLIPTokenizer):
    prompt = [""]
    tokens = tokenizer(prompt, padding="max_length", max_length=config.text_encoder.max_length, return_tensors='np')
    embedding = encoder(tokens)
    return embedding

# TODO: unify data loading in a different file
# TODO: text to image dataset
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


def main(config):
    jax.distributed.initialize()
    devices = jax.devices()
    replication_factor = len(devices)

    key = jax.random.PRNGKey(config.seed)
    logging.info("Random seed:", config.seed)

    # TODO: add drive root param
    save_name = Path("/mnt/disks/persist/checkpoints") / datetime.datetime.now().strftime("text-to-image-checkpoints_%Y-%d-%m_%H-%M-%S")
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
    state = create_train_state(subkey, config, has_context=True)
    save_args = orbax_utils.save_args_from_target(state)

    logging.info(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")

    logging.info("Loading CLIPTokenizer")
    tokenizer = load_tokenizer(config)

    logging.info(f"Loading FlaxCLIPTextModel")
    text_encoder = load_text_encoder(config)

    classifier_free_embedding = compute_classifer_free_embedding(config, text_encoder, tokenizer)

    train_step = build_train_step(config, vqgan=vqgan, train=True, text_encoder=text_encoder, classifier_free_embedding=classifier_free_embedding)
    eval_step = build_train_step(config, vqgan=vqgan, train=False, text_encoder=text_encoder)
    # TODO: param all this
    sample_loop = build_fast_sample_loop(config, vqgan=vqgan, temperature=0.7, proportion=0.5, text_encoder=text_encoder)
    state = flax.jax_utils.replicate(state)

    if config.report_to_wandb:
        wandb.init(project="diffusers-sprint-sundae", config=config)
    else:
        wandb.init(mode="disabled")

    pmap_train_step = jax.pmap(train_step, "replication_axis", in_axes=(0, 0, 0, 0))
    pmap_eval_step = jax.pmap(eval_step, "replication_axis", in_axes=(0, 0, 0, 0))
    pmap_sample_loop = jax.pmap(sample_loop, "replication_axis", in_axes=(0, 0, 0))
    

    step = 0
    while step < config.training.step:
        metrics = dict(loss=0.0, accuracy=0.0)

        pb = tqdm.trange(config.training.batches[0])
        for i in enumerate(pb):
            batch, prompt = next(train_iter)
            batch = einops.rearrange(
                batch.numpy(), "(r b) c h w -> r b c h w", r=replication_factor
            )
            tokens = tokenizer(prompt, padding='max_length', max_length=config.text_encoder.max_length, return_tensors='np')
            tokens = einops.rearrange(
                tokens, "(r b) n -> r b n", r=replication_factor
            )

            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            state, loss, accuracy = pmap_train_step(state, batch, subkeys, conditioning=tokens) # TODO: add donate args, memory save on params
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
            step, flax.jax_utils.unreplicate(state), save_kwargs={"save_args": save_args}
        )

        metrics = dict(loss=0.0, accuracy=0.0)
        pb = tqdm.trange(config.training.batches[1])
        for i in pb:
            batch, prompt = next(eval_iter)
            batch = einops.rearrange(
                batch.numpy(), "(r b) c h w -> r b c h w", r=replication_factor, c=3
            )
            tokens = tokenizer(prompt, padding='max_length', max_length=config.text_encoder.max_length, return_tensors='np')
            tokens = einops.rearrange(
                tokens.numpy(), "(r b) n -> r b n", r=replication_factor
            )

            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            state, loss, accuracy = pmap_eval_step(state, batch, subkeys, conditioning=tokens) # TODO: add donate args, memory save on params
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

        # TODO: how do we want to prompt this?
        # logging.info("sampling from current model")
        # key, subkey = jax.random.split(key)
        # subkeys = jax.random.split(subkey, replication_factor)
        # img = pmap_sample_loop(state.params, subkeys)
        # img = jnp.reshape(img, (-1, config.data.image_size, config.data.image_size, 3))
        # sqrt_num_images = int(sqrt(img.shape[0]))
        # img = custom_to_pil(
        #     np.asarray(
        #         einops.rearrange(
        #             img, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=sqrt_num_images, b2=img.shape[0] // sqrt_num_images
        #         )
        #     )
        # )
        # img.save(Path(save_name) / f"sample-{step:08}.jpg")
        # wandb.log({"sample": wandb.Image(img)}, commit=True, step=step)

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
            batch_size=128,  # TODO: really this shouldn't be under data, it affects the numerics of the model
            num_workers=4,
        ),
        model=dict(
            num_tokens=256,
            dim=1024,
            depth=[2, 12, 2],
            shorten_factor=4,
            resample_type="linear",
            heads=8,
            dim_head=64,
            rotary_emb_dim=32,
            max_seq_len=32,  # effectively squared to 256
            parallel_block=False,
            tied_embedding=False,
            dtype=jnp.bfloat16,  # currently no effect
        ),
        training=dict(
            learning_rate=4e-4,
            unroll_steps=2,
            epochs=100,  # TODO: maybe replace with train steps
            max_grad_norm=1.0,
            weight_decay=1e-2,
        ),
        text_encoder=dict(
            model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            from_pt=True,
            use_fast_tokenizer=True
        ),
        vqgan=dict(name="vq-f8-n256", dtype=jnp.float32),
        jit_enabled=True,
    )

    # Hatman: To eliminate dict_to_namespace
    args = Bunch(
        dict(seed=42)
    )  # if you are changing the seed to get good results, may god help you.
    # args = dict_to_namespace()

    main(dict_to_namespace(config), args)
