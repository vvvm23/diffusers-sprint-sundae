import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import flax.linen as nn
import einops

from typing import Optional

from vqgan_jax.utils import preprocess_vqgan
from sundae import SundaeModel


def build_fast_sample_loop(
    config: dict,
    vqgan: Optional[nn.Module],
    text_encoder: Optional[nn.Module] = None,
    num_samples: int = 4,
    steps: int = 100,
    temperature: float = 1.0,
    proportion: float = 0.5,
):
    def sample_loop(
        params,
        key: jax.random.PRNGKey,
        conditioning: Optional[ArrayLike] = None,
    ):
        model = SundaeModel(config.model)
        assert (conditioning is None) == (text_encoder is None)

        key, subkey = jax.random.split(key)
        x = jax.random.randint(
            subkey,
            (num_samples, config.model.max_seq_len * config.model.max_seq_len),
            0,
            config.model.num_tokens,
        )

        if text_encoder is not None:
            conditioning = text_encoder(conditioning)[0]

        # TODO: figure out how to pass 0dim (temp, proportion) to jit (or does it just work?)
        def sample_step(
            sample: ArrayLike,
            key: jax.random.PRNGKey,
        ):
            key, subkey = jax.random.split(key)
            logits = model.apply({"params": params}, sample, context=conditioning)
            new_sample = jax.random.categorical(subkey, logits / temperature, axis=-1)

            key, subkey = jax.random.split(key)
            mask = jax.random.uniform(subkey, new_sample.shape) > proportion
            new_sample = mask * sample + ~mask * new_sample

            return new_sample, key

        scan_wrapper = lambda _, val: sample_step(*val)

        key, subkey = jax.random.split(key)
        final_sample, _ = jax.lax.fori_loop(0, steps, scan_wrapper, (x, subkey))

        return vqgan.decode_code(final_sample)

    return sample_loop
