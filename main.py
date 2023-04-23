from typing import Callable, Optional, Sequence, Union, Literal

from absl import app, flags, logging

import os
import jax
from clu import platform
from ml_collections import config_flags

FLAGS = flags.FLAGS

OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "An output directory for logs and checkpoints."
)
config_flags.DEFINE_config_file("config", None, "Path to training hyperparameter file.")
# flags.mark_flags_as_required(["config", "output_dir"])
flags.mark_flags_as_required(["config"])



def main(argv):
    del argv

    config = FLAGS.config

    jax.config.update("jax_log_compiles", config.log_compile)
    jax.config.update("jax_debug_nans", config.debug_nans)

    logging.info(f"JAX XLA backend: {FLAGS.jax_xla_backend or 'None'}")
    logging.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    logging.info(f"JAX local devices: {jax.local_devices()}")
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )

    logging.info(f"Config: {config}")

    logging.info(f"Setting Huggingface cache directory to '{config.data.cache_dir}'")
    os.environ['HF_HOME'] = config.data.cache_dir

    from train_unconditional import main as _train_unconditional
    from train_text_to_image import main as _train_text_to_image

    CONFIG_TO_TRAIN_FN = {
        "unconditional": _train_unconditional,
        "text_to_image": _train_text_to_image,
    }


    if config.do_train:
        train_fn = CONFIG_TO_TRAIN_FN[config.train_fn]
        train_fn(config)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
