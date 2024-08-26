# /modeling/sam/prompt_encoder.py

from typing import Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras import layers

from sam2.modeling.sam2_utils import LayerNorm2d
import numpy as np

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

class PromptEncoder(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[layers.Layer] = layers.Activation('gelu'),
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        """
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Point Embeddings
        self.point_embeddings = [
            layers.Embedding(1, embed_dim) for _ in range(4) # pos/neg point + 2 box corners
        ]
        self.not_a_point_embed = layers.Embedding(1, embed_dim)

        # Mask Downscaling 
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = tf.keras.Sequential([
            layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2, padding='same'),
            LayerNorm2d(mask_in_chans // 4),
            activation,
            layers.Conv2D(mask_in_chans, kernel_size=2, strides=2, padding='same'),
            LayerNorm2d(mask_in_chans),
            activation,
            layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='same'),
        ])
        self.no_mask_embed = layers.Embedding(1, embed_dim)

    def get_dense_pe(self) -> tf.Tensor:
        """
        Returns the positional encoding used to encode point prompts.
        """
        return tf.expand_dims(self.pe_layer(self.image_embedding_size), axis=0)

    def _embed_points(
        self,
        points: tf.Tensor,
        labels: tf.Tensor,
        pad: bool,
    ) -> tf.Tensor:
        """Embeds point prompts."""
        points = points + 0.5 # Shift to center of pixel
        if pad:
            padding_point = tf.zeros((tf.shape(points)[0], 1, 2), dtype=tf.float32)
            padding_label = -tf.ones((tf.shape(labels)[0], 1), dtype=tf.int32)
            points = tf.concat([points, padding_point], axis=1)
            labels = tf.concat([labels, padding_label], axis=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # Using one-hot encoding and tf.boolean_mask for conditional embedding addition
        label_conditions = [
            tf.equal(labels, -1),
            tf.equal(labels, 0),
            tf.equal(labels, 1),
            tf.equal(labels, 2),
            tf.equal(labels, 3)
        ]
        embeddings = [
            self.not_a_point_embed.weights[0],
            self.point_embeddings[0].weights[0],
            self.point_embeddings[1].weights[0],
            self.point_embeddings[2].weights[0],
            self.point_embeddings[3].weights[0]
        ]

        for cond, embed in zip(label_conditions, embeddings):
            point_embedding = tf.where(cond, point_embedding + embed, point_embedding)

        return point_embedding

    def _embed_boxes(self, boxes: tf.Tensor) -> tf.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = tf.reshape(boxes, (-1, 2, 2)) 
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding = tf.concat([
            corner_embedding[:, 0:1, :] + self.point_embeddings[2].weights[0], 
            corner_embedding[:, 1:2, :] + self.point_embeddings[3].weights[0]
        ], axis=1)
        return corner_embedding

    def _embed_masks(self, masks: tf.Tensor) -> tf.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[tf.Tensor, tf.Tensor]],
        boxes: Optional[tf.Tensor],
        masks: Optional[tf.Tensor],
    ) -> int:
        """Gets the batch size of the output given the input prompts."""
        if points is not None:
            return tf.shape(points[0])[0] 
        elif boxes is not None:
            return tf.shape(boxes)[0]
        elif masks is not None:
            return tf.shape(masks)[0]
        else:
            return 1

    def call(
        self,
        points: Optional[Tuple[tf.Tensor, tf.Tensor]],
        boxes: Optional[tf.Tensor],
        masks: Optional[tf.Tensor],
        training=False, 
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Embeds different types of prompts.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = tf.zeros((bs, 0, self.embed_dim), dtype=tf.float32)
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = tf.concat([sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = tf.concat([sparse_embeddings, box_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = tf.tile(
                tf.reshape(self.no_mask_embed.weights[0], (1, -1, 1, 1)),
                (bs, 1, self.image_embedding_size[0], self.image_embedding_size[1])
            )

        return sparse_embeddings, dense_embeddings
    
    
    '''
    Explanation:

PositionEmbeddingRandom: This class is now fully implemented. It uses random Gaussian matrices to generate positional encodings, similar to the PyTorch implementation.
Other Components: The rest of the PromptEncoder class, including the _embed_points, _embed_boxes, and _embed_masks methods, remains the same as the previous version.
Key Point:

With PositionEmbeddingRandom implemented, PromptEncoder should now be fully functional for use in the SAM2Base model.'''