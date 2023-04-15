from typing import (
    Callable, 
    Optional, 
    Sequence, 
    Union, 
    Literal
)

from absl import (
    app,
    flags,
    logging
)

from clu import platform

from ml_collections import config_flags


FLAGS = flags.FLAGS

OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "An output directory for logs and checkpoints."
)
config_flags.DEFINE_config_file(
    "config",
    None,
    "Path to training hyperparameter file."
)
flags.mark_flags_as_required(["config", "output_dir"])


def main(argv):
    del argv

    jax.config.update("jax_log_compiles", True)

    logging.info(f"JAX XLA backend: {FLAGS.jax_xla_backend or 'None'}")
    logging.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    logging.info(f"JAX local devices: {jax.local_devices()}")
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )

    logging.info(f"Config: {FLAGS.config}")

    config = FLAGS.config


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
