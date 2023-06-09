from typing import (
    Generator,
    Callable, 
    Optional,
    Sequence,
    Literal,
    Union,
    Tuple,
    Dict
)

import datetime
import functools
from math import sqrt
from pathlib import Path

import numpy as np

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

import flax
import flax.linen as nn
from flax.training import orbax_utils

import orbax.checkpoint

import einops

from absl import logging
import ml_collections as mlc

import tqdm
import wandb

from transformers import (
    FlaxCLIPTextModel,
    CLIPTokenizer,
    CLIPTokenizerFast
)

from torch.utils.data import DataLoader
import torchvision.transforms as T

import vqgan_jax
import vqgan_jax.convert_pt_model_to_jax
from vqgan_jax.utils import custom_to_pil

from train_utils import (
    build_train_step, 
    create_train_state
)
from sample_utils import build_fast_sample_loop
from utils import (
    dict_to_namespace, 
    infinite_loader
)

from datasets import (
    load_dataset,
    Dataset
)


def load_text_encoder(config: mlc.ConfigDict) -> FlaxCLIPTextModel:
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.text_encoder.model_name_or_path, 
        from_pt=config.text_encoder.from_pt
    )
    text_encoder.params = jax.tree_util.tree_map(lambda p: jnp.asarray(p, dtype=jnp.bfloat16), text_encoder.params)
    return text_encoder


def load_tokenizer(config: mlc.ConfigDict) -> Union[CLIPTokenizer, CLIPTokenizerFast]:
    Tokenizer = CLIPTokenizerFast if config.text_encoder.use_fast_tokenizer else CLIPTokenizer
    tokenizer = Tokenizer.from_pretrained(config.text_encoder.model_name_or_path)
    return tokenizer


def tokenize_fn(
    examples: Dict[str, ArrayLike], 
    tokenizer: CLIPTokenizer, 
    captions_column_name: str = "caption",
) -> Dict[str, ArrayLike]:
    captions = examples[captions_column_name]
    inputs = tokenizer(
        captions, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True,
        return_tensors="np",
    )
    examples["input_ids"] = inputs.input_ids

    return examples


def load_datasets(
    config: mlc.ConfigDict, 
    tokenizer: CLIPTokenizer
) -> Tuple[Dataset, Dataset]:
    data_files = {
        "train": config.data.train_file
    }
    if config.data.eval_file:
        data_files["test"] = config.data.eval_file

    datasets = load_dataset(
        config.data.name, 
        data_dir=config.data.train_dir
    )

    if "eval" not in data_files:
        datasets = datasets["train"].train_test_split(
            test_size=config.data.validation_split_percentage / 100,
            seed=config.seed
        )
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]
    train_dataset = train_dataset.remove_columns(['image_id'])
    eval_dataset = eval_dataset.remove_columns(['image_id'])

    _tokenize_fn = functools.partial(
        tokenize_fn,
        tokenizer=tokenizer,
        captions_column_name=config.data.captions_column_name
    )

    train_dataset.set_transform(_tokenize_fn)
    eval_dataset.set_transform(_tokenize_fn)

    train_dataset.with_format("numpy")
    eval_dataset.with_format("numpy")

    return train_dataset, eval_dataset


def collate_fn(examples: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
    output = {}
    output["input_ids"] = np.stack([a['input_ids'] for a in examples], axis=0)
    output["encoding"] = np.stack([a['encoding'] for a in examples], axis=0)

    return output


def create_loader(
    config: mlc.ConfigDict, 
    dataset: Dataset, 
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True
) -> Generator[Dict[str, ArrayLike], None, None]:
    batch_size = batch_size or config.batch_size
    _collate_fn = functools.partial(
        collate_fn,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=_collate_fn,
        drop_last=drop_last
    )
    return infinite_loader(loader)

def compute_classifer_free_embedding(config, encoder: FlaxCLIPTextModel, tokenizer: CLIPTokenizer):
    prompt = [""]
    tokens = tokenizer(prompt, padding="max_length", max_length=config.text_encoder.max_length, return_tensors='np')
    embedding = encoder(tokens['input_ids'])[0]
    return embedding

def main(config: mlc.ConfigDict) -> None:
    jax.distributed.initialize()
    devices = jax.devices()
    replication_factor = len(devices)

    key = jax.random.PRNGKey(config.seed)

    # TODO: add drive root param
    if config.enable_checkpointing:
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

    logging.info(f"Loading tokenizer")
    tokenizer = load_tokenizer(config)

    logging.info(f"Loading dataset")
    train_dataset, eval_dataset = load_datasets(config, tokenizer)
    train_loader = create_loader(
        config,
        train_dataset
    )
    eval_loader = create_loader(
        config,
        eval_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False
    )

    logging.info(f"Loading VQ-GAN")
    vqgan_dtype = getattr(jnp, config.vqgan.dtype)
    vqgan = vqgan_jax.convert_pt_model_to_jax.load_and_download_model(
        config.vqgan.name, dtype=vqgan_dtype
    )
    del vqgan.params['encoder']
    del vqgan.params['quant_conv']

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey, config, has_context=True)
  
    if config.enable_checkpointing:
        save_args = orbax_utils.save_args_from_target(state)

    if config.resume_from_checkpoint:
        state = checkpoint_manager.restore(
            step=config.resume_from_checkpoint_step, 
            items=state,
            directory=config.resume_from_checkpoint_directory
        )
        
    logging.info(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")

    logging.info(f"Loading FlaxCLIPTextModel")
    text_encoder = load_text_encoder(config)

    classifier_free_embedding = compute_classifer_free_embedding(config, text_encoder, tokenizer)
    train_step = build_train_step(config, train=True, text_encoder=text_encoder, classifier_free_embedding=classifier_free_embedding)
    eval_step = build_train_step(config, train=False, text_encoder=text_encoder, classifier_free_embedding=classifier_free_embedding)

    # TODO: param all this, also can we only pass vqgan decoder params?
    num_samples = 1
    sample_prompts = ["An armchair in the shape of an avacado"]*num_samples*replication_factor
    sample_loop = build_fast_sample_loop(config, vqgan=vqgan, temperature=0.7, proportion=0.5, text_encoder=text_encoder, num_samples=num_samples)
    state = flax.jax_utils.replicate(state)

    if config.report_to_wandb:
        wandb.init(project="diffusers-sprint-sundae", config=config)
    else:
        wandb.init(mode="disabled")

    pmap_train_step = jax.pmap(train_step, "replication_axis", in_axes=(0, 0, 0, 0))
    pmap_eval_step = jax.pmap(eval_step, "replication_axis", in_axes=(0, 0, 0, 0))
    pmap_sample_loop = jax.pmap(sample_loop, "replication_axis", in_axes=(0, 0, 0))

    rearrange_fn = functools.partial(
        einops.rearrange, 
        pattern="(d b) n -> d b n",
        d=replication_factor
    )

    step = 0
    while step < config.training.steps:
        metrics = dict(loss=0.0, accuracy=0.0)

        pb = tqdm.trange(config.training.batches[0])
        for i in pb:
            batch = next(train_loader)

            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            state, loss, accuracy = pmap_train_step(
                state, 
                rearrange_fn(batch['encoding']), 
                subkeys, 
                rearrange_fn(batch['input_ids'])
            )
            
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
                step, flax.jax_utils.unreplicate(state), save_kwargs={"save_args": save_args}
            )

        metrics = dict(loss=0.0, accuracy=0.0)
        pb = tqdm.trange(config.training.batches[1])
        for i in pb:
            batch = next(eval_loader)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, replication_factor)
            loss, accuracy = pmap_eval_step(
                state, 
                rearrange_fn(batch['encoding']), 
                subkeys, 
                rearrange_fn(batch['input_ids'])
            )
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
        sample_tokens = tokenizer(sample_prompts, padding='max_length', max_length=config.text_encoder.max_length, return_tensors='np').input_ids
        sample_tokens = rearrange_fn(sample_tokens)
        logging.info("sampling from current model")
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, replication_factor)
        img = pmap_sample_loop(state.params, subkeys, sample_tokens)
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
