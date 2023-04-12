import jax
from jax import lax, numpy as jnp
from jax.typing import ArrayLike

import flax
import flax.linen as nn

from typing import Callable, Optional, Sequence, Union, Literal

import einops
from math import sqrt

from .rotary_embeddings import broadcat, generate_embeddings, apply_rotary_emb

def exists(val):
    return val is not None

# resample functions
class NaiveDownsample(nn.Module):
  shorten_factor: int

  @nn.compact
  def __call__(self, x: ArrayLike):
    x = einops.rearrange(x, 'b (h w) d -> b h w d', h = int(sqrt(x.shape[1])))
    return einops.reduce(x, 'b (h sh) (w sw) d -> b (h w) d', 'mean', sh = self.shorten_factor, sw = self.shorten_factor)

class NaiveUpsample(nn.Module):
  shorten_factor: int

  @nn.compact
  def __call__(self, x: ArrayLike):
    x = einops.rearrange(x, 'b (h w) d -> b h w d', h = int(sqrt(x.shape[1])))
    x = einops.repeat(x, 'b h w d -> b (h sh) (w sw) d', sh=self.shorten_factor, sw=self.shorten_factor)
    return einops.rearrange(x, 'b h w d -> b (h w) d')


class LinearDownsample(nn.Module):
  shorten_factor: int
  
  @nn.compact
  def __call__(self, x: ArrayLike):
    dim = x.shape[-1]
    x = einops.rearrange(x, 'b (h w) d -> b h w d', h = int(sqrt(x.shape[1])))
    x = einops.rearrange(x, 'b (h sh) (w sw) d -> b h w (d sh sw)', sh = self.shorten_factor, sw = self.shorten_factor)
    x = nn.Dense(dim, dtype=jnp.float32)(x)
    return einops.rearrange(x, 'b h w d -> b (h w) d')

class LinearUpsample(nn.Module):
  shorten_factor: int

  @nn.compact
  def __call__(self, x: ArrayLike):
    dim = x.shape[-1] * self.shorten_factor * self.shorten_factor
    x = einops.rearrange(x, 'b (h w) d -> b h w d', h = int(sqrt(x.shape[1])))
    x = nn.Dense(dim, dtype=jnp.float32)(x)
    x = einops.rearrange(x, 'b h w (d sh sw) -> b (h sh) (w sw) d', sh = self.shorten_factor, sw = self.shorten_factor)
    return einops.rearrange(x, 'b h w d -> b (h w) d')

# pre norm wrapper module
class PreNormResidual(nn.Module):
  fn: Callable
  
  @nn.compact
  def __call__(self, x: ArrayLike, **kwargs):
    return self.fn(nn.LayerNorm()(x), **kwargs) + x


# standard transformer ff block
class FeedForward(nn.Module):
  mult: int = 4

  @nn.compact
  def __call__(self, x: ArrayLike):
    dim = x.shape[-1]
    return nn.Sequential([
        nn.Dense(dim * self.mult, dtype=jnp.float32),
        nn.gelu,
        nn.Dense(dim, dtype=jnp.float32)
    ])(x)

# non-causal attention block w/ 2d rotary embeddings
class Attention(nn.Module):
  heads: int = 8
  dim_head: int = 64 

  @nn.compact
  def __call__(self, x: ArrayLike, context: Optional[ArrayLike] = None, pos_emb: Optional[ArrayLike] = None, mask: Optional[ArrayLike] = None):
    h = self.heads
    has_context = context is not None
    dim = h * self.dim_head
    scale = self.dim_head ** -0.5

    q = nn.Dense(dim, use_bias=False, dtype=jnp.float32)(x)
    kv = nn.Dense(2*dim, use_bias=False, dtype=jnp.float32)(context if has_context else x)
    k, v = jnp.split(kv, 2, axis=-1)

    if pos_emb is not None and not has_context:
        q = einops.rearrange(q, 'b (h w) d -> b h w d', h = int(sqrt(q.shape[1])))
        k = einops.rearrange(k, 'b (h w) d -> b h w d', h = int(sqrt(k.shape[1])))
        q, k = apply_rotary_emb(pos_emb, q), apply_rotary_emb(pos_emb, k)
        q = einops.rearrange(q, 'b h w d -> b (h w) d')
        k = einops.rearrange(k, 'b h w d -> b (h w) d')

    q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
    q = q * scale

    sim = jnp.einsum('b h i d, b h j d -> b h i j', q, k)
    mask_value = -jnp.finfo(sim.dtype).max


    if mask is not None:
      mask = einops.rearrange(mask, 'b j -> b () () j')
      jnp.where(~mask, mask_value, sim) # TODO: check mask polarity is right
    # no need for causal mask, model is always non-causal

    attn = nn.softmax(sim, axis=-1)
    out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
    out = einops.rearrange(out, 'b h n d -> b n (h d)', h = h)
    out = nn.Dense(x.shape[-1], use_bias=True, dtype=jnp.float32)(out)

    out = nn.LayerNorm()(out)

    return out

# vanilla transformer module (no recursive elements)
class Transformer(nn.Module):
  depth: int
  heads: int = 8
  dim_head: int = 64
  ff_mult: int = 4
  rotary_emb_dim: Optional[int] = None
  max_seq_len: int = 256
  parallel_block: bool = False 
    
  def setup(self):
    self.layers = [
        (PreNormResidual(Attention(self.heads, self.dim_head)),
        PreNormResidual(FeedForward(self.ff_mult)))
        for i in range(self.depth)
    ]

    self.norm = nn.LayerNorm()

    # TODO: seems to break if we cache here
    # see https://github.com/google/flax/discussions/1921 for potential fix
    # freqs = generate_embeddings(jnp.linspace(-1.0, 1.0, num=self.max_seq_len), self.rotary_emb_dim, max_freq=self.max_seq_len)
    # self.rot_emb = broadcat((freqs[:, None, :], freqs[None, :, :]), axis=-1)

  def __call__(self, x: ArrayLike, context: Optional[ArrayLike] = None, mask: Optional[ArrayLike] = None):
    # TODO: replace with caching
    freqs = generate_embeddings(jnp.linspace(-1.0, 1.0, num=self.max_seq_len), self.rotary_emb_dim, max_freq=self.max_seq_len)
    rot_emb = broadcat((freqs[:, None, :], freqs[None, :, :]), axis=-1)

    for attn, ff in self.layers:
      if self.parallel_block: # gpt-j style arrangement
        x = attn(x, context = context, pos_emb = rot_emb, mask = mask) + ff(x)
      else:
        x = ff(attn(x, context = context, pos_emb = rot_emb, mask = mask))
    
    x = self.norm(x)
    return x


# TODO: check all parameters are passed down!
# (potentially) recursive hourglass transformer
# removes shift + pad ops, no need to non-causal mode
class HourglassTransformer(nn.Module):
  depth: Sequence[int]
  shorten_factor: Union[Sequence[int], int] = 2
  attn_resampling: bool = True
  resample_type: Literal['naive', 'linear'] = 'naive'
  heads: int = 8
  dim_head: int = 64
  rotary_emb_dim: Optional[int] = None
  max_seq_len: int = 256
  parallel_block: bool = False

  def setup(self):
    assert len(self.depth) == 3, 'depth should be tuple of length 3' # TODO: more recursion plz
    pre_layers_depth, valley_depth, post_layers_depth = self.depth

    if isinstance(self.shorten_factor, (tuple, list)):
      shorten_factor, *rest_shorten_factor = self.shorten_factor
    elif isinstance(valley_depth, int): # TODO: bug in OG? why valley_depth?
      shorten_factor, rest_shorten_factor = self.shorten_factor, None
    else:
      shorten_factor, rest_shorten_factor = self.shorten_factor, self.shorten_factor
    
    transformer_kwargs = dict(
        heads = self.heads,
        dim_head = self.dim_head,
        rotary_emb_dim = self.rotary_emb_dim,
        parallel_block = self.parallel_block,
    )

    
    if self.resample_type == 'naive':
      self.downsample = NaiveDownsample(shorten_factor)
      self.upsample = NaiveUpsample(shorten_factor)
    elif self.resample_type == 'linear':
      self.downsample = LinearDownsample(shorten_factor)
      self.upsample = LinearUpsample(shorten_factor)

    assert isinstance(valley_depth, int) or (isinstance(valley_depth, tuple)  and len(valley_depth) == 3), 'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)'
    assert not (isinstance(valley_depth, int) and rest_shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(valley_depth, int):
      self.valley_transformer = Transformer(depth=valley_depth, max_seq_len=self.max_seq_len // shorten_factor, **transformer_kwargs)
    else:
      self.valley_transformer = HourglassTransformer(depth=valley_depth, max_seq_len=self.max_seq_len // (shorten_factor * shorten_factor), resample_type=self.resample_type, shorten_factor=rest_shorten_factor, attn_resampling=attn_resampling, **transformer_kwargs)

    self.attn_resampling_pre_valley = Transformer(depth = 1, max_seq_len=self.max_seq_len // shorten_factor, **transformer_kwargs) if self.attn_resampling else None
    self.attn_resampling_post_valley = Transformer(depth = 1, max_seq_len=self.max_seq_len // shorten_factor, **transformer_kwargs) if self.attn_resampling else None

    self.pre_transformer = Transformer(depth = pre_layers_depth, max_seq_len=self.max_seq_len, **transformer_kwargs)
    self.post_transformer = Transformer(depth = post_layers_depth, max_seq_len=self.max_seq_len, **transformer_kwargs)

  def __call__(self, x: ArrayLike, mask: Optional[ArrayLike] = None):
    s = self.shorten_factor
    b, n = x.shape[:2]
    x = self.pre_transformer(x, mask = mask)
    # TODO: pad x and mask to multiple for pooling, but maybe not needed

    residual = jnp.copy(x)
    downsampled = self.downsample(x)

    if mask is not None:
      downsampled_mask = einops.reduce(mask, 'b (n s) -> b n', 'sum', s=s) > 0
    else:
      downsampled_mask = None

    if self.attn_resampling_pre_valley is not None:
      if mask is not None:
        attn_resampling_mask = einops.rearrange(mask, 'b (n s) -> (b n) s', s = s)
      else:
        attn_resampling_mask = None
 
      downsampled = self.attn_resampling_pre_valley(
          einops.rearrange(downsampled, 'b n d -> (b n) () d'),
          einops.rearrange(x, 'b (n s) d -> (b n) s d', s = s*s),
          mask = attn_resampling_mask
      )
      downsampled = einops.rearrange(downsampled, '(b n) () d -> b n d', b = b)

    x = self.valley_transformer(downsampled, mask = downsampled_mask)
    valley_out = jnp.copy(x)

    x = self.upsample(x)
    x = x + residual

    if exists(self.attn_resampling_post_valley):
        x = self.attn_resampling_post_valley(
            einops.rearrange(x, 'b (n s) d -> (b n) s d', s = s*s),
            einops.rearrange(valley_out, 'b n d -> (b n) () d')
        )

        x = einops.rearrange(x, '(b n) s d -> b (n s) d', b = b)

    # TODO: if we decide to use padding, bring back to original length using `n`

    x = self.post_transformer(x, mask = mask)
    return x
  

# `HourglassTransformer` with embedding layer and token head.
# optionally tie weights
class HourglassTransformerLM(nn.Module):
  num_tokens: int
  dim: int
  depth: Sequence[int]
  shorten_factor: Union[Sequence[int], int] = 2
  attn_resampling: bool = True
  resample_type: Literal['naive', 'linear'] = 'naive'
  heads: int = 8
  dim_head: int = 64
  rotary_emb_dim: Optional[int] = None
  max_seq_len: int = 256
  parallel_block: bool = False
  tied_embedding: bool = False

  @nn.compact
  def __call__(self, x: ArrayLike, mask: Optional[ArrayLike] = None):
    dtype = jnp.float32
    token_embedding = nn.Embed(self.num_tokens, self.dim, dtype=dtype, param_dtype=dtype)
    x = token_embedding(x)
    if self.rotary_emb_dim is None:
      pos_emb = nn.Embed(self.max_seq_len, self.dim, dtype=dtype, param_dtype=dtype)(jnp.arange(x.shape[1]))
      x = x + einops.rearrange(pos_emb, 'n d -> () n d')
    

    x = HourglassTransformer(
        depth = self.depth,
        shorten_factor = self.shorten_factor,
        attn_resampling = self.attn_resampling,
        resample_type = self.resample_type,
        heads = self.heads,
        dim_head = self.dim_head,
        rotary_emb_dim = self.rotary_emb_dim,
        max_seq_len = self.max_seq_len,
        parallel_block = self.parallel_block
    )(x, mask = mask)

    if self.tied_embedding:
      return token_embedding.attend(x)

    return nn.Dense(self.num_tokens)(x)

class SundaeModel(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x: ArrayLike, mask: Optional[ArrayLike] = None):
        config = self.config
        return HourglassTransformerLM(
              num_tokens = config.num_tokens,
              dim = config.dim,
              depth = config.depth,
              shorten_factor = config.shorten_factor,
              resample_type = config.resample_type,
              heads = config.heads,
              dim_head = config.dim_head,
              rotary_emb_dim = config.rotary_emb_dim,
              max_seq_len = config.max_seq_len,
              parallel_block = config.parallel_block,
              tied_embedding = config.tied_embedding,
        )(x, mask=mask)

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    x = jnp.zeros((4, 256), dtype=jnp.int32)
    key, model_key = jax.random.split(jax.random.PRNGKey(0))
    test_model = HourglassTransformerLM(
        num_tokens=1024, 
        dim=512, 
        depth=[2, 4, 2], 
        shorten_factor=2, 
        parallel_block = False,
        tied_embedding = False,
        rotary_emb_dim = 32,
        max_seq_len = 16
    )
    params = test_model.init(model_key, x)
    y = test_model.apply(params, x)
    print('output:', y.shape, y.min(), y.mean(), y.max())
