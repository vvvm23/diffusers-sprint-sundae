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

    model = SundaeModel(config.model)

    if args.checkpoint:
        print(f"Restoring model from {args.checkpoint}")
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        params = ckptr.restore(args.checkpoint, item=None)['params']
    else:
        key, subkey = jax.random.split(key)
        params = model.init(subkey, jnp.zeros((1, config.model.max_seq_len*config.model.max_seq_len), dtype=jnp.int32))['params']

    model.params = params
    samples = model.sample(key=key, num_samples=args.batch_size,
                           steps=args.steps, temperature=args.temperature,
                           min_steps=args.min_steps,
                           proportion=args.proportion,
                           return_history=args.history,
                           progress=True,
                           early_stop=not args.no_early_stop)

    if args.history:
        all_imgs = []
        for t, f in enumerate(samples):
            imgs = vqgan.decode_code(f) # TODO: merge batch and time dimension and encode in one go!
            for i, img in enumerate(imgs):
                custom_to_pil(np.asarray(img)).save(sample_dir / f"sample-{i:02}_{t:03}.png")
    else:
        decoded_samples = vqgan.decode_code(samples)

        for i, img in enumerate(decoded_samples):
            custom_to_pil(np.asarray(img)).save(sample_dir / f"sample-{i:02}.png")

if __name__ == "__main__":
    # TODO: add proper argparsing!: Hatman
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0xFFFF)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--steps", "-n", type=int, default=100)
    parser.add_argument("--temperature", "-t", type=float, default=0.6)
    parser.add_argument("--proportion", "-p", type=float, default=0.3)
    parser.add_argument("--history", action='store_true')
    parser.add_argument("--no-early-stop", action='store_true')
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
