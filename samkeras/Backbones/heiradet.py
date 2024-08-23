# sam2_tfkeras/modeling/backbones/hieradet.py

from functools import partial
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn 

from sam2_tfkeras.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from sam2_tfkeras.modeling.sam2_utils import DropPath, MLP

def do_pool(x: tf.Tensor, pool: layers.Layer, norm: layers.Layer = None) -> tf.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(layers.Layer):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: layers.Layer = None,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = layers.Dense(dim_out * 3)
        self.proj = layers.Dense(dim_out)
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_out // num_heads, dropout=dropout
        )


    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        B, H, W, _ = tf.shape(x)
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = tf.reshape(self.qkv(x), (B, H * W, 3, self.num_heads, -1))
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = tf.unstack(qkv, axis=2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool is not None:
            q = do_pool(tf.reshape(q, (B, H, W, -1)), self.q_pool)
            H, W = tf.shape(q)[1], tf.shape(q)[2]  # downsampled shape
            q = tf.reshape(q, (B, H * W, self.num_heads, -1))

        # TensorFlow's MultiHeadAttention handles heads internally
        attn_output = self.attention_layer(
            query=tf.transpose(q, perm=[1, 0, 2]),
            value=tf.transpose(v, perm=[1, 0, 2]),
            key=tf.transpose(k, perm=[1, 0, 2]),
            attention_mask=None, 
            training=training
        )

        x = tf.transpose(attn_output, perm=[1, 0, 2])  # Transpose back
        x = tf.reshape(x, (B, H, W, -1))
        x = self.proj(x)
        return x


class MultiScaleBlock(layers.Layer):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[layers.Layer, str] = "LayerNormalization",
        q_stride: Tuple[int, int] = None,
        act_layer: layers.Layer = layers.Activation('gelu'),
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = getattr(layers, norm_layer)
            norm_layer = partial(norm_layer, epsilon=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer() 

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride is not None:
            self.pool = layers.MaxPool2D(
                pool_size=q_stride, strides=q_stride, padding='same'
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else layers.Lambda(lambda x: x) 

        self.norm2 = norm_layer()
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = layers.Dense(dim_out) 

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = tf.shape(x)[1], tf.shape(x)[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x, training=training) # Pass training argument
        if self.q_stride is not None:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = tf.shape(shortcut)[1], tf.shape(shortcut)[2]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x, training=training) 
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training) 
        return x


class Hiera(tf.keras.Model):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = self.add_weight(
            shape=(1, embed_dim, *self.window_pos_embed_bkg_spatial_size), 
            initializer='zeros', 
            trainable=True, 
            name='pos_embed'
        )
        self.pos_embed_window = self.add_weight(
            shape=(1, embed_dim, self.window_spec[0], self.window_spec[0]),
            initializer='zeros',
            trainable=True,
            name='pos_embed_window'
        )

        dpr = [
            x.item() for x in tf.linspace(0.0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = [] # List to store blocks in TF

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> tf.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = tf.image.resize(self.pos_embed, size=(h, w), method='bicubic')
        pos_embed = pos_embed + tf.tile(
            window_embed,
            [1, 1, tf.shape(pos_embed)[2] // tf.shape(window_embed)[2], tf.shape(pos_embed)[3] // tf.shape(window_embed)[3]]
        )
        pos_embed = tf.transpose(pos_embed, perm=[0, 2, 3, 1])
        return pos_embed

    def call(self, x: tf.Tensor, training=False) -> List[tf.Tensor]:
        x = self.patch_embed(x) 

        # Add pos embed
        x = x + self._get_pos_embed(tf.shape(x)[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, training=training) 
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = tf.transpose(x, perm=[0, 3, 1, 2])
                outputs.append(feats)

        return outputs
    
    
    '''
    Explanation and Adaptations:

TensorFlow/Keras Layers and Operations:
Uses layers.Dense, layers.LayerNormalization, layers.MaxPool2D, layers.Activation, tf.transpose, tf.reshape, tf.image.resize, tf.tile, etc.
MultiHeadAttention: Leverages layers.MultiHeadAttention from TensorFlow/Keras.
DropPath: You still need to implement the DropPath class (from sam2_utils.py) based on the provided PyTorch code.
Windowing Functions: The window_partition and window_unpartition functions (from backbones/utils.py) are assumed to be adapted to TensorFlow.
MLP Class: Assumes the MLP class (from sam2_utils.py) is available.
Training Argument: The call methods include the training argument for controlling behaviors during training.
Key Points and Potential Challenges:

Windowing Logic: Double-check the padding and windowing logic in MultiScaleBlock to make sure it aligns with the PyTorch implementation.
Hiera Backbone Porting: This is one of the more architecturally complex parts to port. Carefully study the PyTorch code and translate it to TensorFlow/Keras, using the appropriate layers and operations.
'''