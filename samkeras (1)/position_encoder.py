# sam2_tfkeras/modeling/position_encoding.py

import math
from typing import Any, Optional, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

class PositionEmbeddingSine(layers.Layer):
    """
    Positional encoding using RoPE (Rotary Position Embeddings). 
    """
    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = False, 
        scale: Optional[float] = None,
        feat_sizes=(32, 32),  
    ):
        super(PositionEmbeddingSine, self).__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi 
        self.scale = scale

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        self.compute_cis = lambda end_x, end_y: compute_axial_cis(dim=self.num_pos_feats, theta=temperature, end_x=end_x, end_y=end_y)
        self.freqs_cis = self.compute_cis(feat_sizes[0], feat_sizes[1])

    def call(self, x: tf.Tensor, training=False):
        """
        Applies RoPE positional encoding to the input tensor.
        """
        bs, c, h, w = tf.shape(x)  

        # Flatten the spatial dimensions and split into (x, y)
        x = tf.transpose(tf.reshape(x, (bs, c, -1)), perm=[0, 2, 1])  # Shape: (B, H*W, C)
        x = tf.reshape(x, (bs, h, w, c))

        # Apply RoPE using the precomputed freqs_cis
        x, _ = apply_rotary_enc(x, x, self.freqs_cis) 

        # Reshape back to the original format
        x = tf.reshape(x, (bs, h*w, c))  
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, (bs, c, h, w)) 

        return x

class PositionEmbeddingRandom(layers.Layer):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super(PositionEmbeddingRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = self.add_weight(
            shape=(2, num_pos_feats),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=scale),
            trainable=False, 
            name='positional_encoding_gaussian_matrix'
        )

    def _pe_encoding(self, coords: tf.Tensor) -> tf.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = tf.linalg.matmul(coords, self.positional_encoding_gaussian_matrix)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return tf.concat([tf.math.sin(coords), tf.math.cos(coords)], axis=-1)

    def call(self, size: Tuple[int, int]) -> tf.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = tf.ones((h, w), dtype=tf.float32)
        y_embed = tf.math.cumsum(grid, axis=0) - 0.5
        x_embed = tf.math.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / tf.cast(h, tf.float32)
        x_embed = x_embed / tf.cast(w, tf.float32)

        pe = self._pe_encoding(tf.stack([x_embed, y_embed], axis=-1))
        return tf.transpose(pe, perm=[2, 0, 1]) # C x H x W

    def forward_with_coords(
        self, coords_input: tf.Tensor, image_size: Tuple[int, int]
    ) -> tf.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = tf.identity(coords_input) 
        coords = tf.transpose(coords, perm=[0, 2, 1])
        coords = tf.tensor_scatter_nd_update(
            coords, 
            [[0, 0, 0], [0, 1, 0]],
            [coords[0, 0, 0]/ image_size[1], coords[0, 1, 0] / image_size[0]]
        )
        coords = tf.transpose(coords, perm=[0, 2, 1])
        return self._pe_encoding(tf.cast(coords, tf.float32))  # B x N x C

# Rotary Positional Encoding (RoPE) Functions

def init_t_xy(end_x: int, end_y: int):
    t = tf.range(end_x * end_y, dtype=tf.float32)
    t_x = tf.cast(t % end_x, dtype=tf.float32)
    t_y = tf.cast(tf.math.floordiv(t, end_x), dtype=tf.float32)
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (tf.cast(tf.range(0, dim, 4)[: (dim // 4)], tf.float32) / dim))
    freqs_y = 1.0 / (theta ** (tf.cast(tf.range(0, dim, 4)[: (dim // 4)], tf.float32) / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = tf.linalg.matmul(tf.expand_dims(t_x, axis=1), tf.expand_dims(freqs_x, axis=0))
    freqs_y = tf.linalg.matmul(tf.expand_dims(t_y, axis=1), tf.expand_dims(freqs_y, axis=0))
    freqs_cis_x = tf.complex(tf.ones_like(freqs_x), freqs_x)
    freqs_cis_y = tf.complex(tf.ones_like(freqs_y), freqs_y)
    return tf.concat([freqs_cis_x, freqs_cis_y], axis=-1)

def reshape_for_broadcast(freqs_cis: tf.Tensor, x: tf.Tensor):
    ndim = tf.rank(x)
    assert 0 <= 1 < ndim
    tf.debugging.assert_equal(tf.shape(freqs_cis), (tf.shape(x)[-2], tf.shape(x)[-1]))
    shape = [tf.shape(x)[i] if i >= ndim - 2 else 1 for i in range(ndim)]
    return tf.reshape(freqs_cis, shape)

def apply_rotary_enc(
    xq: tf.Tensor,
    xk: tf.Tensor,
    freqs_cis: tf.Tensor,
    repeat_freqs_k: bool = False,
):
    # Split into real and imaginary components 
    xq_ = tf.split(xq, 2, axis=-1) 
    xq_real, xq_imag = tf.squeeze(xq_[0], axis=-1), tf.squeeze(xq_[1], axis=-1)

    xk_ = tf.split(xk, 2, axis=-1) 
    xk_real, xk_imag = tf.squeeze(xk_[0], axis=-1), tf.squeeze(xk_[1], axis=-1)

    freqs_cis_real = tf.math.real(freqs_cis)
    freqs_cis_imag = tf.math.imag(freqs_cis)

    # Apply RoPE rotation 
    xq_out_real = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    xq_out_imag = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real

    xk_out_real = xk_real * freqs_cis_real - xk_imag * freqs_cis_imag
    xk_out_imag = xk_real * freqs_cis_imag + xk_imag * freqs_cis_real

    # Concatenate back the real and imaginary components
    xq_out = tf.concat([tf.expand_dims(xq_out_real, axis=-1), tf.expand_dims(xq_out_imag, axis=-1)], axis=-1)
    xk_out = tf.concat([tf.expand_dims(xk_out_real, axis=-1), tf.expand_dims(xk_out_imag, axis=-1)], axis=-1) 

    return xq_out, xk_out
    
    
