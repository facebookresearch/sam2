# modeling/sam2_utils.py 

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn
import math

def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. 

    Args:
        frame_idx (int): Index of the current frame.
        cond_frame_outputs (dict): Dictionary of outputs from conditioning frames.
        max_cond_frame_num (int): Maximum number of conditioning frames to select.

    Returns:
        tuple: A tuple containing two dictionaries:
            - selected_outputs: Dictionary of selected conditioning frame outputs.
            - unselected_outputs: Dictionary of unselected conditioning frame outputs.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {} 

    assert max_cond_frame_num >= 2, "We should allow using at least 2 conditioning frames."

    selected_outputs = {}

    # The closest conditioning frame before `frame_idx` (if any)
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected_outputs[idx_before] = cond_frame_outputs[idx_before]

    # The closest conditioning frame after `frame_idx` (if any)
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected_outputs[idx_after] = cond_frame_outputs[idx_after]

    # Add other temporally closest conditioning frames until reaching a total
    # of `max_cond_frame_num` conditioning frames.
    num_remain = max_cond_frame_num - len(selected_outputs)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected_outputs),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]
    selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
    unselected_outputs = {
        t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
    }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.

    Args:
        pos_inds (tf.Tensor): Tensor of position indices.
        dim (int): Embedding dimension.
        temperature (int, optional): Temperature for the sine function. Defaults to 10000.

    Returns:
        tf.Tensor: Tensor of positional embeddings.
    """
    pe_dim = dim // 2
    dim_t = tf.range(pe_dim, dtype=tf.float32)
    dim_t = temperature ** (2 * (tf.cast(dim_t // 2, tf.float32)) / pe_dim)

    pos_embed = tf.expand_dims(pos_inds, axis=1) / dim_t
    pos_embed = tf.concat([tf.math.sin(pos_embed), tf.math.cos(pos_embed)], axis=1)
    return pos_embed

def get_activation_fn(activation):
    """
    Return an activation function given a string.

    Args:
        activation (str): Name of the activation function.

    Returns:
        tf.keras.activations: The activation function.
    """
    if activation == "relu":
        return tf.nn.relu
    if activation == "gelu":
        return tf.nn.gelu
    if activation == "glu":
        return tf.nn.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class DropPath(layers.Layer):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=False):
        """
        Applies DropPath to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether the layer is in training mode.

        Returns:
            tf.Tensor: Tensor with DropPath applied.
        """
        if self.drop_prob == 0.0 or not training: 
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (tf.rank(x) - 1)
        random_tensor = tf.keras.backend.random_bernoulli(shape, p=keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor


class MLP(layers.Layer):
    """
    Multi-layer Perceptron (MLP) class.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: layers.Layer = layers.Activation('relu'),
        sigmoid_output: bool = False,
    ) -> None:
        """
        Initializes the MLP.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            activation (layers.Layer, optional): Activation function. Defaults to layers.Activation('relu').
            sigmoid_output (bool, optional): Whether to apply sigmoid activation to the output. 
                Defaults to False.
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

        self.layers = [] 
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(layers.Dense(out_dim)) 
            if i < num_layers - 1:
                self.layers.append(activation)  

    def call(self, x, training=False):
        """
        Forward pass through the MLP.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether the layer is in training mode.

        Returns:
            tf.Tensor: Output tensor.
        """
        for layer in self.layers:
            if isinstance(layer, layers.Activation): 
                x = layer(x)
            else:
                x = layer(x)
        if self.sigmoid_output:
            x = tf.math.sigmoid(x)
        return x


class LayerNorm2d(layers.Layer):
    """
    2D Layer Normalization class.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """
        Initializes the LayerNorm2d layer.

        Args:
            num_channels (int): Number of channels in the input tensor.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super(LayerNorm2d, self).__init__()
        self.weight = self.add_weight(shape=(num_channels,), initializer="ones", trainable=True)
        self.bias = self.add_weight(shape=(num_channels,), initializer="zeros", trainable=True)
        self.eps = eps

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies 2D layer normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tf.Tensor: Normalized tensor with the same shape as the input.
        """
        u = tf.math.reduce_mean(x, axis=1, keepdims=True)
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=1, keepdims=True)
        x = (x - u) / tf.math.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None] 
        return x
