# /modeling/backbones/utils.py

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn 

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = tf.shape(x)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Padding using tf.pad 
    if pad_h > 0 or pad_w > 0:
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    Hp, Wp = H + pad_h, W + pad_w

    x = tf.reshape(x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C))
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), (-1, window_size, window_size, C))
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = tf.shape(windows)[0] // (Hp * Wp // window_size // window_size)
    x = tf.reshape(windows, (B, Hp // window_size, Wp // window_size, window_size, window_size, -1))
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), (B, Hp, Wp, -1))

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  Patch embedding dimension.
        """
        super().__init__()
        self.proj = layers.Conv2D(
            filters=embed_dim, 
            kernel_size=kernel_size, 
            strides=stride, 
            padding='same' if padding else 'valid' # Padding 
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x
    
    '''Explanation and Adaptations:

Padding: Uses tf.pad to handle padding in the window_partition function.
Reshaping and Transposing: Uses tf.reshape and tf.transpose to manipulate the tensor shapes for window partitioning and unpartitioning.
TensorFlow/Keras Layers: Employs layers.Conv2D for the patch embedding operation.
Padding Handling: The padding argument in PatchEmbed is handled to determine whether to use 'same' or 'valid' padding.
Key Points:

The windowing functions (window_partition and window_unpartition) are crucial for the Hiera backbone. Make sure you understand the logic behind them and how they interact with the attention mechanism.
You might need to experiment with different padding strategies in PatchEmbed to find the best match for the original PyTorch behavior.'''