import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional
import einops
from typing import Literal, Optional

import flax.linen as nn
from flax.training import train_state
import optax

import einops

from vqgan_jax.utils import preprocess_vqgan
from sundae import SundaeModel


def cross_entropy(logits, targets):
    logits = einops.rearrange(logits, "b n c -> (b n) c")
    targets = einops.rearrange(targets, "b n -> (b n)")
    nll = jnp.take_along_axis(
        nn.activation.log_softmax(logits, axis=-1),
        jnp.expand_dims(targets, axis=-1),
        axis=-1,
    )
    ce = -jnp.mean(nll)
    return ce


def corrupt_batch(batch, key, num_tokens):
    keys = jax.random.split(key, 3)
    corruption_prob_per_latent = jax.random.uniform(keys[0], (batch.shape[0],))
    rand = jax.random.uniform(keys[1], batch.shape)
    mask = rand < einops.rearrange(corruption_prob_per_latent, "b -> b ()")
    random_idx = jax.random.randint(
        keys[2], batch.shape, 0, num_tokens, dtype=jnp.int32
    )
    return mask * random_idx + ~mask * batch


def create_train_state(key, config: dict, has_context: bool = False):
    model = SundaeModel(config.model)
    params = model.init(
        key,
        jnp.zeros(
            (1, config.model.max_seq_len * config.model.max_seq_len), dtype=jnp.int32
        ),
        context=jnp.zeros((1, config.text_encoder.max_length, config.text_encoder.dim)) if has_context else None
    )["params"]
    lr_scheduler = optax.join_schedules(
        [
            optax.linear_schedule(
                config.training.warmup_start_lr,
                config.training.learning_rate,
                config.training.warmup_steps,
            ),
            optax.linear_schedule(
                config.training.learning_rate,
                config.training.learning_rate / config.training.end_learning_rate_scale,
                config.training.steps - config.training.warmup_steps,
            ),
        ],
        [config.training.warmup_steps],
    )
    opt = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(lr_scheduler, weight_decay=config.training.weight_decay),
    )

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


def build_train_step(
    config: dict,
    vqgan: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    classifier_free_embedding: Optional[ArrayLike] = None,
    train: bool = True,
):
    def train_step(
        state,
        x: ArrayLike,
        key: jax.random.PRNGKey,
        conditioning: Optional[ArrayLike] = None,
    ):
        model = SundaeModel(config.model)
        # if vqgan exists, assume passing image in, so should permute dims and encode.
        # if does not exist, assume x is discrete latents, probably from preprocessing step
        if vqgan is not None:
            x = einops.rearrange(x, "n c h w -> n h w c")
            x = preprocess_vqgan(x)
            _, x = vqgan.encode(x)

        assert (conditioning is None) == (text_encoder is None)

        # if text_encoder exists, assume input `conditioning` is tokens and compute embeddings
        # if doesn't exist, we are probably operating in unconditionally, hence skip.
        # we assume the above as precomputing embeddings is too expensive storage-wise
        # for debugging, just make `text_encoder` a callable that returns a constant
        # note, even when operating in a classifier-free way, we still pass an empty string, and hence a token sequence
        if text_encoder is not None:
            if classifier_free_embedding is not None and config.training.conditioning_dropout > 0.0:
                key, subkey = jax.random.split(key)
                mask = jax.random.uniform(subkey, (conditioning.shape[0],)) < config.training.conditioning_dropout
                mask = einops.rearrange(mask, 'n -> n 1 1')
                conditioning = einops.repeat(classifier_free_embedding, '1 ... -> n ...', n=conditioning.shape[0]) * mask + text_encoder(conditioning)[0] * ~mask
            else:
                conditioning = text_encoder(conditioning)[0]

        def loss_fn(params, key):
            all_logits = []
            key, subkey = jax.random.split(key)
            samples = corrupt_batch(x, subkey, config.model.num_tokens)
            for i in range(
                config.training.unroll_steps
            ):
                key, subkey = jax.random.split(key)
                logits = model.apply({"params": params}, samples, context=conditioning)
                all_logits.append(logits)

                if config.training.temperature > 0.0:
                    samples = jax.random.categorical(
                        subkey, logits / config.training.temperature, axis=-1
                    )
                else:
                    samples = logits.argmax(axis=-1)
                samples = jax.lax.stop_gradient(samples)

            logits = jnp.concatenate(all_logits)
            repeat_batch = jnp.concatenate([x] * config.training.unroll_steps)
            total_loss = cross_entropy(logits, repeat_batch)
            total_accuracy = (logits.argmax(axis=-1) == repeat_batch).mean()

            return total_loss, 100.0 * total_accuracy

        if train:
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, key
            )
            grads = jax.lax.pmean(grads, "replication_axis")
            new_state = state.apply_gradients(grads=grads)

            return new_state, loss, accuracy

        loss, accuracy = loss_fn(state.params, key)
        return loss, accuracy

    return train_step
