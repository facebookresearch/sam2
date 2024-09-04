# modeling/memory_attention.py

import tensorflow as tf
from tensorflow.keras import layers

from .sam.transformer import RoPEAttention  # Assuming RoPEAttention will be defined in transformer.py 
from .sam2_utils import get_activation_fn 

class MemoryAttentionLayer(layers.Layer):
    """
    A single layer of Memory Attention, consisting of:
        - Self-attention within the target (current frame) features
        - Cross-attention from target features to memory features
        - An MLP applied to the target features.
    """

    def __init__(
        self,
        activation: str,  # Name of the activation function for the MLP
        cross_attention: layers.Layer,  # The cross-attention layer (e.g., RoPEAttention)
        d_model: int,  # Dimension of the model (embedding dimension)
        dim_feedforward: int,  # Dimension of the feedforward network in the MLP
        dropout: float,  # Dropout rate
        pos_enc_at_attn: bool,  # Whether to add positional encoding to self-attention
        pos_enc_at_cross_attn_keys: bool,  # Whether to add pos. enc. to cross-attn keys
        pos_enc_at_cross_attn_queries: bool,  # Whether to add pos. enc. to cross-attn queries
        self_attention: layers.Layer,  # The self-attention layer (e.g., Attention)
    ):
        super(MemoryAttentionLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout

        # Initialize the attention layers
        self.self_attn = self_attention 
        self.cross_attn_image = cross_attention 

        # Initialize the MLP layers
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)

        # Initialize Layer Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        # Initialize Dropout layers
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.activation = get_activation_fn(activation)  # Get the activation function

        # Set positional encoding flags 
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def call(self, tgt, memory, pos=None, query_pos=None, training=False): 
        """
        Forward pass through the MemoryAttentionLayer.

        Args:
            tgt (tf.Tensor): Target (current frame) features.
            memory (tf.Tensor): Memory features.
            pos (tf.Tensor, optional): Positional encodings for memory features. Defaults to None.
            query_pos (tf.Tensor, optional): Positional encodings for target features. Defaults to None.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Processed target features. 
        """
        # --- Self-Attention ---
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2 # Add positional encodings if enabled
        tgt2 = self.self_attn(q, k, value=tgt2, training=training) 
        tgt = tgt + self.dropout1(tgt2, training=training) 

        # --- Cross-Attention ---
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2, 
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory, 
            training=training 
        )
        tgt = tgt + self.dropout2(tgt2, training=training)

        # --- MLP ---
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2)), training=training))
        tgt = tgt + self.dropout3(tgt2, training=training)

        return tgt

class MemoryAttention(layers.Layer):
    """
    Memory Attention module, consisting of multiple MemoryAttentionLayers. 
    This module is used to condition the current frame's features on the features 
    from previous frames (memory). 
    """
    def __init__(
        self,
        d_model: int, # Dimension of the model (embedding dimension)
        pos_enc_at_input: bool,  # Whether to add positional encoding at the input
        layer: layers.Layer, # A single MemoryAttentionLayer
        num_layers: int, # The number of MemoryAttentionLayers
        batch_first: bool = True, # Whether the batch dimension comes first in the input tensors
    ):
        super(MemoryAttention, self).__init__()
        self.d_model = d_model 
        self.layers = [layer for _ in range(num_layers)] # Create a list of MemoryAttentionLayers
        self.num_layers = num_layers
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pos_enc_at_input = pos_enc_at_input 
        self.batch_first = batch_first 

    def call(
        self,
        curr: tf.Tensor,   # Features of the current frame
        memory: tf.Tensor, # Features from the memory 
        curr_pos: Optional[tf.Tensor] = None, # Positional encodings for the current frame features
        memory_pos: Optional[tf.Tensor] = None,  # Positional encodings for the memory features
        num_obj_ptr_tokens: int = 0,  # Number of object pointer tokens (if used)
        training=False 
    ):
        """
        Forward pass through the MemoryAttention module.

        Args:
            curr (tf.Tensor): Features of the current frame.
            memory (tf.Tensor): Features from the memory. 
            curr_pos (tf.Tensor, optional): Positional encodings for the current frame features. Defaults to None.
            memory_pos (tf.Tensor, optional): Positional encodings for the memory features. Defaults to None.
            num_obj_ptr_tokens (int, optional): The number of object pointer tokens (if used). Defaults to 0.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: The output features of the current frame, conditioned on the memory. 
        """
        if isinstance(curr, list): 
            # If 'curr' is a list, assume it's a list with a single element 
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1 
            curr, curr_pos = curr[0], curr_pos[0] 

        tf.debugging.assert_equal(
            tf.shape(curr)[1], tf.shape(memory)[1],
            message="Batch size must be the same for 'curr' and 'memory'."
        )

        output = curr 

        # Add positional encoding at the input if enabled
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # If 'batch_first' is True, transpose the tensors to (B, T, C) format
        if self.batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])
            curr_pos = tf.transpose(curr_pos, perm=[1, 0, 2]) 
            memory = tf.transpose(memory, perm=[1, 0, 2])
            memory_pos = tf.transpose(memory_pos, perm=[1, 0, 2]) 

        # Pass the inputs through each MemoryAttentionLayer
        for layer in self.layers: 
            output = layer(
                tgt=output, 
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                training=training
            )

        # Apply Layer Normalization to the output
        normed_output = self.norm(output) 

        # Transpose the output back to (T, B, C) format if 'batch_first' is True
        if self.batch_first: 
            normed_output = tf.transpose(normed_output, perm=[1, 0, 2]) 
            curr_pos = tf.transpose(curr_pos, perm=[1, 0, 2])

        # Return the normalized output
        return normed_output
