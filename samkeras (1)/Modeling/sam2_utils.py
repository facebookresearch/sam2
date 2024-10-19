# modeling/sam2_utils.py

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn 
import math

def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs` 
    that are temporally closest to the current frame at `frame_idx`. 

    This function is used in the `SAM2Base` model to select the most relevant 
    conditioning frames for the memory attention mechanism.

    Args:
        frame_idx (int): Index of the current frame.
        cond_frame_outputs (dict): A dictionary containing the outputs of the 
            conditioning frames, where the keys are frame indices and the values 
            are dictionaries of frame outputs.
        max_cond_frame_num (int): The maximum number of conditioning frames 
            to select. If -1, all conditioning frames will be selected.

    Returns:
        tuple: A tuple containing two dictionaries:
            - selected_outputs (dict): A dictionary containing the outputs of the 
                selected conditioning frames.
            - unselected_outputs (dict): A dictionary containing the outputs of the
                unselected conditioning frames.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        # Select all conditioning frames if max_cond_frame_num is -1 or if the 
        # number of conditioning frames is less than or equal to max_cond_frame_num
        return cond_frame_outputs, {}  

    assert max_cond_frame_num >= 2, "At least 2 conditioning frames should be allowed."

    selected_outputs = {}

    # Select the closest conditioning frame before the current frame
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected_outputs[idx_before] = cond_frame_outputs[idx_before]

    # Select the closest conditioning frame after the current frame
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected_outputs[idx_after] = cond_frame_outputs[idx_after]

    # Select additional conditioning frames based on their temporal proximity
    # to the current frame, until reaching max_cond_frame_num
    num_remain = max_cond_frame_num - len(selected_outputs)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected_outputs),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]

    # Add the selected frames to the selected_outputs dictionary
    selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)

    # Create a dictionary of unselected conditioning frames
    unselected_outputs = {
        t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
    }

    # Return the selected and unselected conditioning frame outputs
    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Generate 1D sinusoidal positional encodings. This function is based on 
    the positional encoding method from the "Attention is All You Need" paper.

    Args:
        pos_inds (tf.Tensor): A tensor of position indices.
        dim (int): The dimension of the positional encodings.
        temperature (int, optional): The temperature value used to scale the 
            positional encodings. Defaults to 10000.

    Returns:
        tf.Tensor: A tensor of sinusoidal positional encodings.
    """
    pe_dim = dim // 2 
    dim_t = tf.range(pe_dim, dtype=tf.float32) 
    dim_t = temperature ** (2 * (tf.cast(dim_t // 2, tf.float32)) / pe_dim) 

    pos_embed = tf.expand_dims(pos_inds, axis=1) / dim_t
    pos_embed = tf.concat([tf.math.sin(pos_embed), tf.math.cos(pos_embed)], axis=1) 
    return pos_embed

def get_activation_fn(activation):
    """
    Return a TensorFlow activation function based on its name. 

    Args:
        activation (str): The name of the activation function ("relu", "gelu", or "glu").

    Returns:
        tf.keras.activations: The corresponding TensorFlow activation function.
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
    Drop paths (Stochastic Depth) per sample  during training.
    This helps prevent overfitting by randomly dropping residual paths in the network.
    Adapted from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        """
        Initializes the DropPath layer.

        Args:
            drop_prob (float, optional): The probability of dropping a path. Defaults to 0.0.
            scale_by_keep (bool, optional): Whether to scale the output by (1 - drop_prob). Defaults to True.
        """
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
        shape = (tf.shape(x)[0],) + (1,) * (tf.rank(x) - 1)  # Shape for the random tensor
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
        input_dim: int,  # Input dimension
        hidden_dim: int,  # Hidden dimension
        output_dim: int,  # Output dimension
        num_layers: int,  # Number of layers
        activation: layers.Layer = layers.Activation('relu'), # Activation function
        sigmoid_output: bool = False, # Whether to apply sigmoid to the output 
    ) -> None:
        """
        Initializes the MLP.
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
    This applies layer normalization over the channel dimension of a 4D tensor. 
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """
        Initializes the LayerNorm2d layer.

        Args:
            num_channels (int): The number of channels in the input tensor.
            eps (float, optional): A small value added to the variance to avoid 
                division by zero. Defaults to 1e-6.
        """
        super(LayerNorm2d, self).__init__()
        self.weight = self.add_weight(shape=(num_channels,), initializer="ones", trainable=True)
        self.bias = self.add_weight(shape=(num_channels,), initializer="zeros", trainable=True)
        self.eps = eps

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies 2D layer normalization to the input tensor.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            tf.Tensor: The normalized tensor with the same shape as the input.
        """
        u = tf.math.reduce_mean(x, axis=1, keepdims=True)  # Mean over the channel dimension
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=1, keepdims=True) # Variance over the channel dimension
        x = (x - u) / tf.math.sqrt(s + self.eps) # Normalize
        x = self.weight[:, None, None] * x + self.bias[:, None, None] # Scale and shift
        return x
