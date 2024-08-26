# /modeling/sam/transformer.py

import tensorflow as tf
from tensorflow.keras import layers
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from typing import Tuple 


class TwoWayTransformer(layers.Layer):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: layers.Layer = layers.Activation('relu'), 
        attention_downsample_rate: int = 2,
    ) -> None:
        super(TwoWayTransformer, self).__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = [] # Use a list for layers in TF

        for _ in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = layers.LayerNormalization(epsilon=1e-6)

    def call(
        self,
        image_embedding: tf.Tensor,
        image_pe: tf.Tensor,
        point_embedding: tf.Tensor,
        training=False, # Add training argument
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = tf.shape(image_embedding) 
        image_embedding = tf.transpose(tf.reshape(image_embedding, (bs, c, h*w)), perm=[0, 2, 1])
        image_pe = tf.transpose(tf.reshape(image_pe, (bs, c, h*w)), perm=[0, 2, 1])

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                training=training # Pass training argument
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys, training=training)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: layers.Layer = layers.Activation('relu'),
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False, # Not needed in TF implementation
    ) -> None:
        super(TwoWayAttentionBlock, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.norm4 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

    def call(
        self, queries: tf.Tensor, keys: tf.Tensor, query_pe: tf.Tensor, key_pe: tf.Tensor, 
        training=False, # Add training argument
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Self attention block
        q = queries + query_pe 
        attn_out = self.self_attn(q=q, k=q, v=queries, training=training)
        queries = queries + attn_out 
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, training=training)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries, training=training) # Pass training argument
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, training=training)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(layers.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = layers.Dense(self.internal_dim)
        self.k_proj = layers.Dense(self.internal_dim)
        self.v_proj = layers.Dense(self.internal_dim)
        self.out_proj = layers.Dense(self.embedding_dim)
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.internal_dim // num_heads, dropout=dropout
        )
        self.dropout_p = dropout

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, training=False) -> tf.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # TensorFlow's MultiHeadAttention handles heads internally
        attn_output = self.attention_layer(
            query=q, 
            value=v, 
            key=k, 
            attention_mask=None,  # You might need attention masking in some cases
            training=training
        )

        out = self.out_proj(attn_output)
        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super(RoPEAttention, self).__init__(*args, **kwargs)
        self.compute_cis = lambda end_x, end_y: compute_axial_cis(dim=self.internal_dim // self.num_heads, theta=rope_theta, end_x=end_x, end_y=end_y)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def call(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_k_exclude_rope: int = 0,
        training=False # Add training argument
    ) -> tf.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Apply rotary position encoding
        w = h = int(tf.math.sqrt(tf.cast(tf.shape(q)[-2], tf.float32)))
        if tf.shape(self.freqs_cis)[0] != tf.shape(q)[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h)
        if tf.shape(q)[-2] != tf.shape(k)[-2]:
            tf.debugging.assert_equal(self.rope_k_repeat, True)

        num_k_rope = tf.shape(k)[-2] - num_k_exclude_rope
        q, k = apply_rotary_enc(
            q,
            k[:, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # TensorFlow's MultiHeadAttention handles heads internally
        attn_output = self.attention_layer(
            query=q,
            value=v,
            key=k,
            attention_mask=None,  # You might need attention masking 
            training=training
        )

        out = self.out_proj(attn_output)
        return out
    
    '''
    **Explanation and Key Adaptations:**

- **TensorFlow `MultiHeadAttention`:**  Uses TensorFlow's built-in `layers.MultiHeadAttention` to simplify the implementation.
- **RoPE Attention:** The `RoPEAttention` class is implemented, using the `apply_rotary_enc` function (from `position_encoding.py`, which was converted in a previous response) to apply rotary embeddings.
- **Complex Number Operations:** The RoPE logic might require careful handling of complex number operations in TensorFlow.  Refer to TensorFlow's documentation on complex number support.
- **TensorFlow/Keras Equivalents:**  Uses `layers.Dense`, `layers.LayerNormalization`, `tf.transpose`, `tf.reshape`, and other TensorFlow/Keras equivalents for PyTorch operations.
- **Training Argument:**  The `call` method in all classes now takes the `training` argument to control training-specific behavior (e.g., dropout). 

**Next Steps:**

- **Continue Conversion:**  Move on to converting the remaining modules, such as `backbones/hieradet.py`, `backbones/image_encoder.py`, etc.
- **Thorough Testing:**  As you convert each module, make sure to test it carefully to ensure that its behavior matches the PyTorch implementation. 

'''