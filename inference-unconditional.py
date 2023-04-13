import jax
from jax import lax, numpy as jnp
from jax.typing import ArrayLike

import flax
import flax.linen as nn

from flax.training import checkpoints

from typing import Callable, Optional, Sequence, Union, Literal

import numpy as np

import datetime
from pathlib import Path

import tqdm

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax
from vqgan_jax.utils import custom_to_pil

from utils import dict_to_namespace
from sundae import SundaeModel
import argparse


def setup_sample_dir(root: str = "samples", prompt: Optional[str] = None):
    sample_root = Path(root)
    sample_root.mkdir(exist_ok=True)

    if prompt is not None:
        prompt_dir = sample_root / prompt
    else:
        prompt_dir = sample_root

    prompt_dir.mkdir(exist_ok=True)

    sample_dir = prompt_dir / datetime.datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    sample_dir.mkdir(exist_ok=True)

    return sample_dir


def main(config, args):
    print("Config:", config)

    print("JAX devices:", jax.devices())
    key = jax.random.PRNGKey(args.seed)
    print("Random seed:", args.seed)

    print(f"Loading VQ-GAN")
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
        config.vqgan.name, dtype=config.vqgan.dtype
    )

    key, subkey = jax.random.split(key)

    sample_dir = setup_sample_dir()

    print(f"Restoring model from {args.checkpoint}")
    state_restored = checkpoints.restore_checkpoint(
        ckpt_dir=args.checkpoint, target=None, step=args.checkpoint_step
    )
    print(state_restored)
    params = state_restored.params
    del state_restored

    model = SundaeModel(config.model)

    key, subkey = jax.random.split(key)
    sample = jax.random.randint(
        subkey,
        (args.batch_size, config.model.max_seq_len * config.model.max_seq_len),
        0,
        config.model.num_tokens,
        dtype=jnp.int32,
    )

    print("Beginning sampling loop")
    # TODO: jit loop properly. don't naively jit loop as compile time will scale with sample steps
    for i in tqdm.trange(args.sample_steps):
        logits = model.apply({"params": params}, sample)

        key, subkey = jax.random.split(key)
        new_sample = jax.random.categorical(
            subkey, logits / args.sample_temperature, axis=-1
        )

        key, subkey = jax.random.split(key)

        # approx x% of mask will be False (aka where to update)
        mask = jax.random.uniform(subkey, new_sample.shape) > args.sample_proportion

        # where True (aka, where to fix) copy from original sample
        new_sample[mask] = sample[mask]

        if jnp.all(new_sample == sample):
            print(f"No change during sampling step {i}. Terminating.")
            break

        # now copy back to original sample to update
        sample = new_sample

    print("Decoding latents with VQGAN")
    decoded_samples = vqgan.decode_code(sample)

    print(f"Saving to {sample_dir.as_posix()}")
    for i, img in enumerate(decoded_samples):
        custom_to_pil(np.asarray(img)).save(sample_dir / f"sample-{i}.png")


if __name__ == "__main__":
    # TODO: add proper argparsing!: Hatman
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoint-step", type=int, default=None, help="`None` loads latest."
    )
    parser.add_argument("--seed", type=int, default=0xFFFF)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-proportion", type=float, default=0.5)
    args = parser.parse_args()

    config = dict(
        model=dict(
            num_tokens=16_384,
            dim=1024,
            depth=[2, 10, 2],
            shorten_factor=4,
            resample_type="linear",
            heads=8,
            dim_head=64,
            rotary_emb_dim=32,
            max_seq_len=16,
            parallel_block=True,
            tied_embedding=False,
            dtype=jnp.bfloat16,
        ),
        vqgan=dict(name="vq-f16", dtype=jnp.bfloat16),
        jit_enabled=True,
    )

    main(dict_to_namespace(config), args)
