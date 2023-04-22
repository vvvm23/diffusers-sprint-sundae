import ml_collections as mlc

import jax.numpy as jnp


def get_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()

    config.train_fn = "text_to_image"
    config.batch_size = 32
    config.seed = 0
    config.do_train = True

    config.data = dict(
        name="parquet",
        num_workers=4,
        train_dir="",
        validation_dir="",
        train_file="",
        eval_file="",
        captions_column_name="caption",
        overwrite_cache=False,
        flip_p=0.5,
        image_size=256,
        max_train_samples=-1,
        max_eval_samples=-1,
        validation_split_percentage=10,
        preprocessing_num_workers=1
    )
    config.model = dict(
        model_name_or_path="sundae-default",
        config_name="sundae-default",
        num_tokens=16_384,
        dim=1024,
        depth=[3, 12, 3],
        shorten_factor=4,
        resample_type="linear",
        heads=8,
        dim_head=64,
        rotary_emb_dim=32,
        max_seq_len=32,  # effectively squared to 256
        parallel_block=False,
        tied_embedding=False,
        dtype="bfloat16",
    )
    config.text_encoder = dict(
        model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        from_pt=True,
        use_fast_tokenizer=True,
        max_length = 77,
    )
    config.training = dict(
        learning_rate=4e-4,
        unroll_steps=3,
        steps=1_000_000,
        warmup_steps=4000,
        warmup_start_lr=1e-6,
        end_learning_rate=3e-6,
        max_grad_norm=5.0,
        weight_decay=0.0,
        temperature=1.0,
        batches=(4000, 200),
        conditioning_dropout = 0.2
    )
    config.vqgan = dict(name="vq-f8", dtype="bfloat16")

    config.checkpoint = dict(keep_period=100, max_to_keep=3)

    config.logging_dir = "./logs"
    config.report_to_wandb = True

    config.debug_nans = False
    config.log_compile = False

    return config
