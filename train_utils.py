import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional
import einops

import flax.linen as nn
from flax.training import train_state
import optax

import einops

from vqgan_jax.utils import preprocess_vqgan
from sundae import SundaeModel


def corrupt_batch(batch, key, num_tokens):
    keys = jax.random.split(key, 3)
    corruption_prob_per_latent = jax.random.uniform(keys[0], (batch.shape[0],))
    rand = jax.random.uniform(keys[1], batch.shape)
    mask = rand < einops.rearrange(corruption_prob_per_latent, "b -> b ()")
    random_idx = jax.random.randint(keys[2], batch.shape, 0, num_tokens, dtype=jnp.int32)
    return mask * random_idx + ~mask * batch


def create_train_state(key, config: dict):
    model = SundaeModel(config.model)
    params = model.init(
        key,
        jnp.zeros(
            [1, config.model.max_seq_len * config.model.max_seq_len], dtype=jnp.int32
        ),
    )["params"]
    opt = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(config.training.learning_rate, weight_decay=config.training.weight_decay)
        # optax.lamb(config.training.learning_rate, weight_decay=config.training.weight_decay)
        # optax.sgd(config.training.learning_rate)
    ) 

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


def build_train_step(config: dict, vqgan: Optional[nn.Module] = None, text_encoder: Optional[nn.Module] = None, train: bool = True):
    def train_step(state, x: ArrayLike, key: jax.random.PRNGKey, conditioning: Optional[ArrayLike] = None):
        model = SundaeModel(config.model)
        # if vqgan exists, assume passing image in, so should permute dims and encode.
        # if does not exist, assume x is discrete latents, probably from preprocessing step
        if vqgan is not None: 
            x = einops.rearrange(x, "n c h w -> n h w c")
            x = preprocess_vqgan(x)
            x = jnp.asarray(x, dtype=jnp.int32)
            _, x = vqgan.encode(x)
            x_copy = x.copy()

        assert (conditioning is None) == (text_encoder is None)

        # if text_encoder exists, assume input `conditioning` is tokens and compute embeddings
        # if doesn't exist, we are probably operating in unconditionally, hence skip.
        # we assume the above as precomputing embeddings is too expensive storage-wise
        # for debugging, just make `text_encoder` a callable that returns a constant
        # note, even when operating in a classifier-free way, we still pass an empty string, and hence a token sequence
        if text_encoder is not None:
            conditioning = text_encoder(conditioning)

        # TODO: classifier-free guidance, how to do in best way? Does it work on SUNDAE?
        # maybe precompute empty embedding and compile as a constant?

        def loss_fn(params, key):
            all_logits = []
            key, subkey = jax.random.split(key)
            samples = corrupt_batch(x, subkey, config.model.num_tokens)
            for i in range(config.training.unroll_steps): # TODO: replace with real jax loop, otherwise compile time scales with num iters.
                key, subkey = jax.random.split(key)
                logits = model.apply({"params": params}, samples, context=conditioning)
                all_logits.append(logits)

                if i != config.training.unroll_steps - 1:
                    # samples = jax.random.categorical(subkey, logits, axis=-1)
                    samples = logits.argmax(axis=-1) # TODO: cool idea, low temperature sampling as a regularisation
                    samples = jax.lax.stop_gradient(samples)

            # total_loss = jnp.concatenate(losses).mean()
            logits = jnp.concatenate(all_logits)
            repeat_batch = jnp.concatenate([x]*config.training.unroll_steps)
            total_loss = optax.softmax_cross_entropy_with_integer_labels(logits, repeat_batch).mean()
            total_accuracy = (logits.argmax(axis=-1) == repeat_batch).mean()

            return total_loss, 100.0 * total_accuracy

        if train:
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, key
            )
            grads = jax.lax.pmean(grads, 'replication_axis')
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss, accuracy

        loss, accuracy = loss_fn(state.params, key)
        return loss, accuracy


    # if config.jit_enabled: # we pmap by default so this does nothing now
    #     return jax.jit(train_step)
    return train_step
