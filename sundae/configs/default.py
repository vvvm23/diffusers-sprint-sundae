import ml_collections as mlc

import jax.numpy as jnp


def get_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()

    config.seed = 0

    config.data = dict(
        name="ffhq256",
        batch_size=48,  # TODO: really this shouldn't be under data, it affects the numerics of the model
        num_workers=4,
    )
    config.model = dict(
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
    )
    config.training = dict(
        learning_rate=1e-4,
        unroll_steps=2,
        epochs=100,  # TODO: maybe replace with train steps
    )
    config.vqgan = dict(
        name="vq-f16", 
        dtype=jnp.bfloat16
    )

    config.jit_enabled = True

    return config
