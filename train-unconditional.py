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
from utils import dict_to_namespace

# Hatman: added imports for wandb, argparse, and bunch
import wandb
import argparse
from bunch import Bunch


# TODO: expand for whatever datasets we will use
# TODO: auto train-valid split
def get_data_loader(
    name: Literal["ffhq256"], batch_size: int = 1, num_workers: int = 0, train: bool = True
):
    if name in ["ffhq256"]:
        dataset = ImageFolder(
            "data/ffhq256",
            # transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
            transform=T.Compose([T.ToTensor()]),
        )
    else:
        raise ValueError(f"unrecognised dataset name '{name}'")

    if train:
        dataset = Subset(dataset, list(range(60_000)))
    else:
        dataset = Subset(dataset, list(range(60_000, 70_000)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        shuffle=True
    )
    return dataset, loader


def main(config, args):
    print("Config:", config)
    print("Args:", args)

    # import jax
    # from sundae import SundaeModel
    # from jax import make_jaxpr
    # x = jnp.zeros((1, config.model.max_seq_len*config.model.max_seq_len), dtype=jnp.int32)
    # model = SundaeModel(config.model)
    # params = model.init(jax.random.PRNGKey(0), x)['params']
    # print(make_jaxpr(lambda t: model.apply({'params': params}, t))(x))
    # exit()

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
        config.data.name, config.data.batch_size*2, config.data.num_workers, train=False
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
    train_step = build_train_step(config, vqgan=vqgan, train=True)
    eval_step = build_train_step(config, vqgan=vqgan, train=False)
    state = flax.jax_utils.replicate(state)

    # TODO: wandb logging plz: Hatman
    # TODO: need flag to toggle on and off otherwise we will pollute project
    # wandb.init(project="diffusers-sprint-sundae", config=config)
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_opts = orbax.checkpoint.CheckpointManagerOptions(keep_period=10, max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(save_name, orbax_checkpointer, checkpoint_opts)
    save_args = orbax_utils.save_args_from_target(state)

    pmap_train_step = jax.pmap(train_step, 'replication_axis', in_axes=(0, 0, 0))
    pmap_eval_step = jax.pmap(eval_step, 'replication_axis', in_axes=(0, 0, 0))

    log_interval = 64
    sample_count = 0
    from vqgan_jax.utils import custom_to_pil
    for ei in range(config.training.epochs):
        total_loss = 0.0
        total_accuracy = 0.0

        wandb_metrics = dict(loss=0.0, accuracy=0.0)
        pb = tqdm.tqdm(train_loader)
        # TODO: need to change these functions to not exhaust loader before going to next phase
        for i, (batch, _) in enumerate(pb):
            batch = einops.rearrange(batch.numpy(), '(r b) c h w -> r b c h w', r = replication_factor, c = 3)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            state, loss, accuracy = pmap_train_step(state, batch, subkeys) # TODO: add donate args, memory save on params
            loss, accuracy = loss.mean(), accuracy.mean()
            total_loss += loss
            total_accuracy += accuracy

            wandb_metrics['loss'] += loss
            wandb_metrics['accuracy'] += accuracy

            pb.set_description(
                f"[epoch {ei+1}] [train] loss: {total_loss / (i+1):.6f}, accuracy {total_accuracy / (i+1):.2f}"
            )

        checkpoint_manager.save(ei, flax.jax_utils.unreplicate(state), save_kwargs={'save_args': save_args})

        # TODO: need to change these functions to not exhaust loader before going to next phase
        total_loss = 0.0
        total_accuracy = 0.0
        pb = tqdm.tqdm(eval_loader)
        for i, (batch, _) in enumerate(pb):
            batch = einops.rearrange(batch.numpy(), '(r b) c h w -> r b c h w', r = replication_factor, c = 3)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            loss, accuracy = pmap_eval_step(state, batch, subkeys)

            loss, accuracy = loss.mean(), accuracy.mean()
            total_loss += loss
            total_accuracy += accuracy

            # TODO: need to separate out train and eval time metrics
            wandb_metrics['loss'] += loss
            wandb_metrics['accuracy'] += accuracy

            pb.set_description(
                f"[epoch {ei+1}] [eval] loss: {total_loss / (i+1):.6f}, accuracy {total_accuracy / (i+1):.2f}"
            )
        
        # TODO: pjit sample?
        print("sampling from current model")
        key, subkey = jax.random.split(key)
        # TODO: param all this
        sample = model.sample(subkey, num_samples=4, steps=100, temperature=0.7, proportion=0.4, early_stop=False, progress=False, return_history=False) # TODO: param this
        decoded_image = jax.jit(vqgan.decode_code)(sample)
        custom_to_pil(einops.rearrange(decoded_image, "(b1 b2) h w c -> (b1 h) (b2 w) c"), b1=2, b2=2).save(save_name / f'sample-{sample_count:04}.jpg')
        sample_count += 1


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
            batch_size=32,  # TODO: really this shouldn't be under data, it affects the numerics of the model
            num_workers=4,
        ),
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
        training=dict(
            learning_rate = 3e-4,
            unroll_steps=2,
            epochs=100, # TODO: maybe replace with train steps
            max_grad_norm=5.0,
            weight_decay=1e-2,
            temperature=0.0
        ),
        vqgan=dict(name="vq-f8-n256", dtype=jnp.bfloat16),
        jit_enabled=True, # TODO: remove, pmap will already jit function
    )

    # Hatman: To eliminate dict_to_namespace
    args = Bunch(dict(seed=42)) # if you are changing the seed to get good results, may god help you.
    # args = dict_to_namespace()

    main(dict_to_namespace(config), args)
