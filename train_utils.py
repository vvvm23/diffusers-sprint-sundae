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
        _, batch = vqgan(batch)  # TODO: we only need to encode!!!

        def loss_fn(params, key):
            all_logits = []
            key, subkey = jax.random.split(key)
            samples = corrupt_batch(batch, subkey, config.model.num_tokens)
            for i in range(config.training.unroll_steps): # TODO: replace with real jax loop, otherwise compile time scales with num iters.
                samples = jax.lax.stop_gradient(samples)
                key, subkey = jax.random.split(key)
                logits = model.apply({"params": params}, samples)
                all_logits.append(logits)

                # total_accuracy = (
                    # total_accuracy + (logits.argmax(axis=-1) == batch).mean()
                # )

                if i != config.training.unroll_steps - 1:
                    samples = jax.random.categorical(subkey, logits, axis=-1)

            # total_loss = jnp.concatenate(losses).mean()
            logits = jnp.concatenate(all_logits)
            repeat_batch = jnp.concatenate([batch]*config.training.unroll_steps)
            total_loss = optax.softmax_cross_entropy_with_integer_labels(logits, repeat_batch).mean()
            total_accuracy = (logits.argmax(axis=-1) == repeat_batch).mean()

            return total_loss, 100.0 * total_accuracy

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, key
        )
        grads = jax.lax.pmean(grads, 'replication_axis')
        state = state.apply_gradients(grads=grads)

        return state, loss, accuracy

    if config.jit_enabled:
        return jax.jit(train_step)
    return train_step
