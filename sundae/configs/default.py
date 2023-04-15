import ml_collections as mlc

import jax.numpy as jnp


def get_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()

    config.do_train = False
    config.batch_size = 48
    config.seed = 0

    config.data = dict(
        name="ffhq256",
        num_workers=4,
        image_size=224,
        train_dir="",
        validation_dir=""
    )
    config.model = dict(
        model_name_or_path="",
        config_name="",
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
        dtype="float16",
        cache_dir=None,
        use_auth_token=None,
    )
    config.training = dict(
        learning_rate=1e-4,
        unroll_steps=2,
        num_epochs=100,
        max_steps=None,
        warmup_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.0
    )
    config.vqgan = dict(
        name="vq-f16", 
        dtype=jnp.bfloat16
    )

    config.jit_enabled = True

    config.logging_dir = "./logs"
    config.report_to = [
        "tensorboard"
    ]
    
    return config
