# sam2keras/modeling/memory_attention.py

import tensorflow as tf
from tensorflow.keras import layers
from sam2.modeling.sam.transformer import RoPEAttention  
from sam2.modeling.sam2_utils import get_activation_fn

class MemoryAttentionLayer(layers.Layer):
    def __init__(
        self,
        activation: str,
        cross_attention: layers.Layer,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: layers.Layer,
    ):
        super(MemoryAttentionLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def call(self, tgt, memory, pos=None, query_pos=None, training=False): 
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, training=training) 
        tgt = tgt + self.dropout1(tgt2, training=training)

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            training=training,
        )
        tgt = tgt + self.dropout2(tgt2, training=training)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2)), training=training))
        tgt = tgt + self.dropout3(tgt2, training=training)
        return tgt


class MemoryAttention(layers.Layer):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: layers.Layer,
        num_layers: int,
        batch_first: bool = True,  
    ):
        super(MemoryAttention, self).__init__()
        self.d_model = d_model
        self.layers = [layer for _ in range(num_layers)] 
        self.num_layers = num_layers
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def call(
        self,
        curr: tf.Tensor,  
        memory: tf.Tensor,
        curr_pos: Optional[tf.Tensor] = None, 
        memory_pos: Optional[tf.Tensor] = None,
        num_obj_ptr_tokens: int = 0, 
        training=False
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        tf.debugging.assert_equal(
            tf.shape(curr)[1], tf.shape(memory)[1], 
            message="Batch size must be the same for curr and memory"
        )

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first 
            output = tf.transpose(output, perm=[1, 0, 2])
            curr_pos = tf.transpose(curr_pos, perm=[1, 0, 2])
            memory = tf.transpose(memory, perm=[1, 0, 2])
            memory_pos = tf.transpose(memory_pos, perm=[1, 0, 2])

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_obj_ptr_tokens=num_obj_ptr_tokens, 
                training=training
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = tf.transpose(normed_output, perm=[1, 0, 2])
            curr_pos = tf.transpose(curr_pos, perm=[1, 0, 2])

        return normed_output
    
    '''
    TensorFlow Attention: Uses layers.MultiHeadAttention for both self-attention and cross-attention.
RoPE Attention (RoPEAttention) is imported but still needs a separate implementation, which we will provide later.
TensorFlow/Keras Equivalents: Leverages TensorFlow/Keras layers and functions (like layers.Dense, layers.Dropout, layers.LayerNormalization, tf.transpose, and tf.debugging.assert_equal).
Simplified Logic: The num_k_exclude_rope parameter is removed as the exclusion logic will be handled within the RoPEAttention layer.
Training Argument: The call method in both classes now includes the training argument to control dropout and other training-specific behaviors.'''
    
    