# ported from lucidrains implementation https://github.com/lucidrains/rotary-embedding-torch
from math import pi, log

import torch

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from einops import rearrange, repeat

# helper functions
def exists(val):
    return val is not None

def broadcat(tensors, axis = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    axis = (axis + shape_len) if axis < 0 else axis
    axes = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(axes) if i != axis]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(axis, (axis, axes[axis]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    #tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    tensors = list(map(lambda t: jnp.broadcast_to(t[0], t[1]), zip(tensors, expandable_shapes)))
    return jnp.concatenate(tensors, axis = axis)

# rotary embedding helper functions
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x[..., 0], x[..., 1]
    
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    return jnp.concatenate((t_left, t, t_right), axis = -1)

def generate_embeddings(t, dim: int, max_freq: int = 16):
    freqs = jnp.linspace(1., max_freq / 2, dim // 2) * pi
    freqs = jnp.einsum('..., f -> ... f', t, freqs)
    freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
    return freqs
