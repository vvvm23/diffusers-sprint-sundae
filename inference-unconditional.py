import jax
from jax import lax, numpy as jnp
from jax.typing import ArrayLike
from jax import make_jaxpr

import flax
import flax.linen as nn

import orbax.checkpoint

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
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    params = ckptr.restore(args.checkpoint, item=None)['params']

    model = SundaeModel(config.model)

    key, subkey = jax.random.split(key)
    sample = jax.random.randint(
        subkey,
        (args.batch_size, config.model.max_seq_len * config.model.max_seq_len),
        0,
        config.model.num_tokens,
        dtype=jnp.int32,
    )

    #@jax.jit
    def jit_sample(sample, key):
        logits = model.apply({"params": params}, sample)

        key, subkey = jax.random.split(key)
        # new_sample = logits.argmax(axis=-1)
        new_sample = jax.random.categorical(
            subkey, logits / args.sample_temperature, axis=-1
        )

        # approx x% of mask will be False (aka where to update)
        mask = jax.random.uniform(key, new_sample.shape) > args.sample_proportion

        # where True (aka, where to fix) copy from original sample
        # new_sample[mask] = sample[mask]
        # new_sample = new_sample.at[mask].set(sample.at[mask]) # <~~ no bueno!
        new_sample = mask * sample + ~mask * new_sample # JIT-compile must have static shape, hence this monstrosity

        return new_sample

    print("Beginning sampling loop")
    history = []
    # TODO: jit loop properly. don't naively jit loop as compile time will scale with sample steps
    for i in tqdm.trange(args.sample_steps):
        key, subkey = jax.random.split(key)
        new_sample = jit_sample(sample, subkey)

        if jnp.all(new_sample == sample):
            print(f"No change during sampling step {i}. Terminating.")
            break

        # now copy back to original sample to update
        sample = new_sample
        if args.history:
            history.append(sample)

    if args.history:
        all_imgs = []
        for t, f in enumerate(history):
            for i, s in enumerate(f):
                img = vqgan.decode_code(s)[0]
                custom_to_pil(np.asarray(img)).save(sample_dir / f"sample-{i}_{j}.png")
    else:
        decoded_samples = vqgan.decode_code(sample)

        for i, img in enumerate(decoded_samples):
            custom_to_pil(np.asarray(img)).save(sample_dir / f"sample-{i}.png")

if __name__ == "__main__":
    # TODO: add proper argparsing!: Hatman
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0xFFFF)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-temperature", type=float, default=0.6)
    parser.add_argument("--sample-proportion", type=float, default=0.3)
    parser.add_argument("--history", action='store_true')
    args = parser.parse_args()

    config = dict(
        model=dict(
            num_tokens=256,
            # num_tokens=10,
            dim=1024,
            # dim=32,
            depth=[2, 12, 2],
            # depth=[1,1,1],
            shorten_factor=4,
            resample_type="linear",
            heads=2,
            dim_head=64,
            # dim_head=8,
            rotary_emb_dim=32,
            # rotary_emb_dim=4,
            max_seq_len=32, # effectively squared to 256
            # max_seq_len=4, # effectively squared to 256
            parallel_block=True,
            tied_embedding=False,
            dtype=jnp.bfloat16, # currently no effect
        ),
        vqgan=dict(name="vq-f8-n256", dtype=jnp.bfloat16),
        jit_enabled=True,
    )

    main(dict_to_namespace(config), args)
