# sam2_tfkeras/modeling/sam2_utils.py

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn 
import copy
import math

def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. 
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
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
    """
    pe_dim = dim // 2
    dim_t = tf.range(pe_dim, dtype=tf.float32)
    dim_t = temperature ** (2 * (tf.cast(dim_t // 2, tf.float32)) / pe_dim)

    pos_embed = tf.expand_dims(pos_inds, axis=1) / dim_t
    pos_embed = tf.concat([tf.math.sin(pos_embed), tf.math.cos(pos_embed)], axis=1)
    return pos_embed

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tf.nn.relu
    if activation == "gelu":
        return tf.nn.gelu
    if activation == "glu":
        return tf.nn.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# 'get_clones' is not needed in TensorFlow/Keras

class DropPath(layers.Layer):
    # Adapted from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=False):
        if self.drop_prob == 0.0 or not training: # Use 'training' argument
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (tf.rank(x) - 1)
        random_tensor = tf.keras.backend.random_bernoulli(shape, p=keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor


class MLP(layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: layers.Layer = layers.Activation('relu'), 
        sigmoid_output: bool = False,
    ) -> None:
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

        self.layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(layers.Dense(out_dim)) # Add Dense layers
            if i < num_layers - 1: 
                self.layers.append(activation) # Add activation in between

    def call(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, layers.Activation): # Apply activations conditionally
                x = layer(x)
            else:
                x = layer(x)
        if self.sigmoid_output:
            x = tf.math.sigmoid(x)
        return x


class LayerNorm2d(layers.Layer):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super(LayerNorm2d, self).__init__()
        self.weight = self.add_weight(shape=(num_channels,), initializer="ones", trainable=True)
        self.bias = self.add_weight(shape=(num_channels,), initializer="zeros", trainable=True)
        self.eps = eps

    def call(self, x: tf.Tensor) -> tf.Tensor:
        u = tf.math.reduce_mean(x, axis=1, keepdims=True)
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=1, keepdims=True)
        x = (x - u) / tf.math.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None] 
        return x
    
    '''
    Explanation and Adaptations:

select_closest_cond_frames and get_1d_sine_pe: These functions are directly translated with TensorFlow equivalents for PyTorch operations.
DropPath: The PyTorch DropPath is adapted using tf.keras.backend.random_bernoulli and the training argument in the call method.
MLP: The MLP class is implemented using a list of layers.Dense and activation layers (layers.Activation).
LayerNorm2d: The custom 2D layer normalization is implemented using TensorFlow's tf.math.reduce_mean, tf.math.square, and tf.math.sqrt.
get_clones: This function is not needed in TensorFlow/Keras.
Key Points:

The DropPath implementation uses the training argument to control dropout, so it is important to pass this argument appropriately when using this layer.
You now have all the necessary building blocks for the MemoryEncoder, MemoryAttentionLayer, and MaskDecoder modules.'''