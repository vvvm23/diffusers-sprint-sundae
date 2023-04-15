
from typing import (
    Any,
    Callable,
    Optional
)

import logging

import torch
import torchvision
import torchvision.transforms as transforms

import jax
import jax.numpy as jnp
import optax

import transformers
from transformers import (
    is_tensorboard_available,
    set_seed,
    AutoConfig,
    FlaxAutoModelForImageClassification,
)


Model = Any
Config = Any
Dataset = Any
TensorboardWriter = Any


logger = logging.getLogger(__name__)


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def load_datasets(config) -> Tuple[Dataset, Dataset]:
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        config.data.train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(config.data.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    eval_dataset = torchvision.datasets.ImageFolder(
        config.data.validation_dir,
        transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.CenterCrop(config.data.image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset, eval_dataset


def create_loader(dataset, config, batch_size, for_training=True) -> torch.utils.data.DataLoader:
    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])

        batch = {"pixel_values": pixel_values, "labels": labels}
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=for_training,
        num_workers=config.data.num_workers,
        persistent_workers=True,
        drop_last=for_training,
        collate_fn=collate_fn,
    )
    return data_loader


def load_model_and_config(config, num_labels=None) -> Tuple[Model, Config]:
    if config.model.config_name:
        model_config = AutoConfig.from_pretrained(
            config.model.config_name,
            num_labels=num_labels,
            image_size=config.data.image_size,
            cache_dir=config.model.cache_dir,
            use_auth_token=True if config.model.use_auth_token else None,
        )
    elif config.model.model_name_or_path:
        model_config = AutoConfig.from_pretrained(
            config.model.model_name_or_path,
            num_labels=num_labels,
            image_size=config.data.image_size,
            cache_dir=config.model.cache_dir,
            use_auth_token=True if config.model.use_auth_token else None,
        )
    else:
        model_config = CONFIG_MAPPING[config.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if config.model.model_name_or_path:
        model = FlaxAutoModelForImageClassification.from_pretrained(
            config.model.model_name_or_path,
            config=config,
            seed=config.seed,
            dtype=getattr(jnp, config.model.dtype),
            use_auth_token=True if config.model.use_auth_token else None,
        )
    else:
        model = FlaxAutoModelForImageClassification.from_config(
            model_config,
            seed=config.seed,
            dtype=getattr(jnp, config.model.dtype),
        )

    return model, model_config


def do_tensorboard_logging(config) -> bool:
    return (
        is_tensorboard_available()
        and "tensorboard" in config.report_to
        and jax.process_index() == 0
    )


def create_tensorboard_writer(config) -> Optional[TensorboardWriter]:
    summary_writer = None
    if do_tensorboard_logging(config):
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(config.logging_dir))
        except ImportError as e:
            logger.warning(
                "Unable to display metrics through TensorBoard"
                f" because some package are not installed: {e}"
            )
    return summary_writer


def train_fn(config):
    set_seed(config.seed)

    train_dataset, eval_dataset = load_datasets(config)
    model, model_config = load_model_and_config(config, num_labels=len(train_dataset.classes))

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    train_loader = create_loader(train_dataset, config, train_batch_size, for_training=True)
    eval_loader = create_loader(eval_dataset, config, eval_batch_size, for_training=False)

    tensorboard_writer = create_tensorboard_writer(config)

    rng = jax.random.PRNGKey(config.seed)
    rng, dropout_rng = jax.random.split(rng)

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        config.training.num_epochs,
        config.training.warmup_steps,
        config.training.learning_rate,
    )
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=config.training.adam_beta1,
        b2=config.training.adam_beta2,
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )

    state = TrainState.create(
        apply_fn=model.__call__, 
        params=model.params, 
        tx=adamw, 
        dropout_rng=dropout_rng
    )

    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        return loss.mean()

    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
        metrics = {"loss": loss, "accuracy": accuracy}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        steps_per_epoch = len(train_dataset) // train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_loader:
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

        train_time += time.time() - train_start

        train_metric = unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_steps = len(eval_dataset) // eval_batch_size
        eval_step_progress_bar = tqdm(total=eval_steps, desc="Evaluating...", position=2, leave=False)
        for batch in eval_loader:
            # Model forward
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, batch, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

            eval_step_progress_bar.update(1)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

        # Print metrics and update progress bar
        eval_step_progress_bar.close()
        desc = (
            f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {round(eval_metrics['loss'].item(), 4)} | "
            f"Eval Accuracy: {round(eval_metrics['accuracy'].item(), 4)})"
        )
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if do_tensorboard_logging(config) == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(config.output_dir, params=params)
