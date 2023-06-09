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
        train_dir="/mnt/disks/persist/laion_en_aesthetic_vqganf8_encoded/",
        validation_dir="",
        train_file="/mnt/disks/persist/laion_en_aesthetic_vqganf8_encoded/",
        eval_file="",
        captions_column_name="caption",
        overwrite_cache=False,
        flip_p=0.5,
        image_size=256,
        max_train_samples=-1,
        max_eval_samples=-1,
        validation_split_percentage=0.1,
        preprocessing_num_workers=0,
        cache_dir="/mnt/disks/persist/huggingface_cache",
    )
    config.model = dict(
        model_name_or_path="sundae-default",
        config_name="text_to_image_default",
        num_tokens=16_384,
        dim=1024,
        depth='(2, 16, 2)',
        shorten_factor=4,
        resample_type="linear",
        heads=8,
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
        dim=1024,
    )
    config.training = dict(
        learning_rate=1e-4,
        unroll_steps=2,
        steps=1_800_000,
        warmup_steps=18_000,
        warmup_start_lr=1e-6,
        end_learning_rate_scale=100,
        max_grad_norm=10.0,
        weight_decay=0.0,
        temperature=1.0,
        batches=(5000, 400),
        conditioning_dropout = 0.1
    )
    config.vqgan = dict(name="vq-f8", dtype="bfloat16")

    config.checkpoint = dict(keep_period=100, max_to_keep=10)

    config.logging_dir = "./logs"
    config.report_to_wandb = True

    config.debug_nans = False
    config.log_compile = False
    config.enable_checkpointing = True
    config.resume_from_checkpoint = False
    config.resume_from_checkpoint_step = -1
    config.resume_from_checkpoint_directory = ""

    return config
