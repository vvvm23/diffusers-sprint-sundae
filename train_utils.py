import jax
import jax.numpy as jnp
import einops

import flax.linen as nn
import optax

import einops

from vqgan_jax.utils import preprocess_vqgan
from sundae import SundaeModel


def corrupt_batch(batch, key, num_tokens):
    keys = jax.random.split(key, 3)
    corruption_prob_per_latent = jax.random.uniform(keys[0], (batch.shape[0],))
    rand = jax.random.uniform(keys[1], batch.shape)
    mask = rand < einops.rearrange(corruption_prob_per_latent, "b -> b ()")
    random_idx = jax.random.randint(keys[2], batch.shape, 0, num_tokens)
    return mask * random_idx + ~mask * batch


def build_train_step(config: dict, vqgan: nn.Module):
    def train_step(state, batch, key):
        model = SundaeModel(config.model)
        batch = einops.rearrange(batch, "n c h w -> n h w c")
        batch = preprocess_vqgan(batch)
        _, batch = vqgan(batch)

        def loss_fn(params, key):
            losses = []
            total_accuracy = 0.0
            key, subkey = jax.random.split(key)
            samples = corrupt_batch(batch, subkey, config.model.num_tokens)
            for i in range(config.training.unroll_steps):
                samples = jax.lax.stop_gradient(samples)
                key, subkey = jax.random.split(key)
                logits = model.apply({"params": params}, samples)

                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
                losses.append(loss)
                total_accuracy = (
                    total_accuracy + (logits.argmax(axis=-1) == batch).mean()
                )

                if i != config.training.unroll_steps - 1:
                    samples = jax.random.categorical(subkey, logits, axis=-1)

            total_loss = jnp.concatenate(losses).mean()
            return total_loss, 100.0 * total_accuracy / config.training.unroll_steps

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, key
        )
        state = state.apply_gradients(grads=grads)

        return state, loss, accuracy

    if config.jit_enabled:
        return jax.jit(train_step)
    return train_step
