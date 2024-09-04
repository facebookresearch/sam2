# sam/transformer.py

import tensorflow as tf
from tensorflow.keras import layers
from ..position_encoding import apply_rotary_enc, compute_axial_cis
from ..sam2_utils import MLP, get_activation_fn
import math

class Attention(layers.Layer):
    """
    An attention layer that allows for downsampling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,  # Embedding dimension
        num_heads: int,  # Number of attention heads
        downsample_rate: int = 1,  # Downsampling rate
        dropout: float = 0.0,  # Dropout rate
        kv_in_dim: int = None,  # Input dimension for keys and values (if different)
    ) -> None:
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        # Linear projections for queries, keys, and values
        self.q_proj = layers.Dense(self.internal_dim)
        self.k_proj = layers.Dense(self.internal_dim)
        self.v_proj = layers.Dense(self.internal_dim)
        self.out_proj = layers.Dense(self.embedding_dim)

        # Multi-Head Attention Layer
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.internal_dim // num_heads, dropout=dropout
        )
        self.dropout_p = dropout

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, training=False) -> tf.Tensor:
        """
        Forward pass through the Attention layer.
        """
        # Project the queries, keys, and values
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Apply Multi-Head Attention 
        attn_output = self.attention_layer(
            query=q,
            value=v,
            key=k,
            attention_mask=None, 
            training=training
        )

        # Project the output back to the original dimension
        out = self.out_proj(attn_output)
        return out


class RoPEAttention(Attention):
    """
    Attention layer with Rotary Positional Embeddings (RoPE).
    """
    def __init__(
        self,
        *args,
        rope_theta=10000.0, 
        rope_k_repeat=False, 
        feat_sizes=(32, 32),  
        **kwargs,
    ):
        super(RoPEAttention, self).__init__(*args, **kwargs)
        self.compute_cis = lambda end_x, end_y: compute_axial_cis(
            dim=self.internal_dim // self.num_heads, theta=rope_theta, end_x=end_x, end_y=end_y
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1]) 
        self.freqs_cis = freqs_cis 
        self.rope_k_repeat = rope_k_repeat 

    def call(
        self, 
        q: tf.Tensor, 
        k: tf.Tensor, 
        v: tf.Tensor, 
        num_k_exclude_rope: int = 0,
        training=False 
    ) -> tf.Tensor:
        # Project the queries, keys, and values
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Apply RoPE 
        w = h = int(tf.math.sqrt(tf.cast(tf.shape(q)[-2], tf.float32))) 

        # Recompute RoPE frequencies if the sequence length has changed
        if tf.shape(self.freqs_cis)[0] != tf.shape(q)[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h) 

        # Apply rotary positional encodings
        q, k = apply_rotary_enc(q, k, freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat) 

        # Apply Multi-Head Attention 
        attn_output = self.attention_layer(
            query=q,
            value=v,
            key=k,
            attention_mask=None,
            training=training
        )

        # Project the output
        out = self.out_proj(attn_output)
        return out

class TwoWayTransformer(layers.Layer):
    """
    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.
    """
    def __init__(
        self,
        depth: int,  # Number of layers in the transformer
        embedding_dim: int,  # Channel dimension for the input embeddings
        num_heads: int,  # Number of heads for multihead attention
        mlp_dim: int,  # Channel dimension internal to the MLP block
        activation: layers.Layer = layers.Activation('relu'),  # Activation to use in the MLP
        attention_downsample_rate: int = 2,  # Downsample rate for the attention
    ) -> None:
        """
        Initializes the TwoWayTransformer.
        """
        super(TwoWayTransformer, self).__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = []  # Use a list to store transformer layers

        # Create 'depth' number of TwoWayAttentionBlock layers
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),  # Skip PE in the first layer
                )
            )

        # Final attention layer from tokens to image
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # Layer Normalization after the final attention
        self.norm_final_attn = layers.LayerNormalization(epsilon=1e-6)

    def call(
        self,
        image_embedding: tf.Tensor,  # Image embeddings (B, C, H, W)
        image_pe: tf.Tensor,  # Positional encodings for the image (B, C, H, W)
        point_embedding: tf.Tensor,  # Point embeddings (B, N, C)
        training=False, 
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through the TwoWayTransformer.
        """

        # Reshape image embeddings from (B, C, H, W) to (B, H*W, C)
        bs, c, h, w = tf.shape(image_embedding) 
        image_embedding = tf.transpose(tf.reshape(image_embedding, (bs, c, h*w)), perm=[0, 2, 1]) 
        image_pe = tf.transpose(tf.reshape(image_pe, (bs, c, h*w)), perm=[0, 2, 1])

        # Prepare queries (point embeddings) and keys (image embeddings)
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                training=training
            )

        # Apply final attention from points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys, training=training) 
        queries = queries + attn_out
        queries = self.norm_final_attn(queries) 

        return queries, keys


class TwoWayAttentionBlock(layers.Layer):
    """
    A transformer block with two-way attention:
        - Self-attention within the queries
        - Cross-attention from queries to keys
        - Cross-attention from keys to queries
        - MLP applied to the queries
    """

    def __init__(
        self,
        embedding_dim: int,  # Embedding dimension
        num_heads: int,  # Number of attention heads
        mlp_dim: int = 2048,  # Dimension of the MLP
        activation: layers.Layer = layers.Activation('relu'),  # Activation function
        attention_downsample_rate: int = 2,  # Downsampling rate for attention
        skip_first_layer_pe: bool = False,  # Whether to skip positional encoding in the first layer
    ) -> None:
        """
        Initializes the TwoWayAttentionBlock.
        """
        super(TwoWayAttentionBlock, self).__init__()

        # Attention layers
        self.self_attn = Attention(embedding_dim, num_heads) 
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # Layer Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.norm4 = layers.LayerNormalization(epsilon=1e-6)

        # MLP 
        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def call(
        self, 
        queries: tf.Tensor, # Query tensor
        keys: tf.Tensor,  # Key tensor
        query_pe: tf.Tensor,  # Positional encoding for queries
        key_pe: tf.Tensor,  # Positional encoding for keys
        training=False  
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through the TwoWayAttentionBlock.

        Args:
            queries (tf.Tensor): Query tensor.
            keys (tf.Tensor): Key tensor.
            query_pe (tf.Tensor): Positional encoding for queries.
            key_pe (tf.Tensor): Positional encoding for keys.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the processed queries and keys.
        """

        # --- Self-attention ---
        if self.skip_first_layer_pe: 
            queries = self.self_attn(q=queries, k=queries, v=queries, training=training) 
        else:
            q = queries + query_pe 
            attn_out = self.self_attn(q=q, k=q, v=queries, training=training)
            queries = queries + attn_out 
        queries = self.norm1(queries) 

        # --- Cross-attention from queries to keys ---
        q = queries + query_pe 
        k = keys + key_pe 
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, training=training) 
        queries = queries + attn_out 
        queries = self.norm2(queries) 

        # --- MLP ---
        mlp_out = self.mlp(queries, training=training) 
        queries = queries + mlp_out 
        queries = self.norm3(queries)

        # --- Cross-attention from keys to queries ---
        q = queries + query_pe 
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, training=training)
        keys = keys + attn_out
        keys = self.norm4(keys) 

        return queries, keys 
