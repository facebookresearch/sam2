# modeling/sam/mask_decoder.py

'''
Explanation of Boolean Flags:
use_high_res_features (Default: False)
True: The MaskDecoder will use higher-resolution feature maps from the image encoder (if available) to improve the quality of the mask predictions. This is often beneficial but can increase computation.
False: The decoder will only use the lowest-resolution feature map from the encoder.
iou_prediction_use_sigmoid (Default: False)
True: A sigmoid activation will be applied to the output of the IoU prediction head to ensure the IoU values are between 0 and 1.
False: The IoU predictions will be raw logits (not constrained to 0-1).
dynamic_multimask_via_stability (Default: False)
True: During inference, the MaskDecoder will dynamically choose between the single mask output token and the best multimask output token based on the stability score. This helps improve robustness when dealing with ambiguous prompts.
False: The decoder will always use the output token specified by multimask_output in the call method.
pred_obj_scores (Default: False)
True: The model will predict object scores in addition to masks. This can be useful for filtering out low-confidence mask predictions.
False: The model will only predict masks.
pred_obj_scores_mlp (Default: False)
True: An MLP (Multi-Layer Perceptron) will be used to predict the object scores. This can potentially improve accuracy but adds more parameters to the model.
False: A simple linear layer will be used to predict object scores.
use_multimask_token_for_obj_ptr (Default: False)
True: The model will use the output tokens from the multimask predictions to calculate object pointers.
False: The model will use the single mask output token for object pointer calculation.'''

from typing import List, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras import layers

from ..sam2_utils import LayerNorm2d, MLP
from .transformer import TwoWayTransformer
from ncps.tf import CfC 

class MaskDecoder(layers.Layer):
    """
    Predicts masks given an image and prompt embeddings.

    This implementation uses a hybrid approach, combining the original 
    TwoWayTransformer from SAM with a CfC (Closed-form Continuous-time) 
    layer from the `ncps` library to enhance temporal modeling. 
    """
    def __init__(
        self,
        *,
        transformer_dim: int,  # Channel dimension of the transformer
        transformer: layers.Layer,  # TensorFlow transformer layer
        num_multimask_outputs: int = 3,  # Number of masks to predict for disambiguation
        activation: Type[layers.Layer] = layers.Activation('gelu'),  # Activation function
        iou_head_depth: int = 3,  # Depth of the MLP for mask quality prediction
        iou_head_hidden_dim: int = 256,  # Hidden dimension of the MLP for mask quality
        use_high_res_features: bool = False,  # Use high-resolution features in the decoder 
        iou_prediction_use_sigmoid=False,  # Use sigmoid on the output of IoU prediction head 
        dynamic_multimask_via_stability=False,  # Dynamic multimask selection
        dynamic_multimask_stability_delta=0.05,  # Delta used when computing the stability score
        dynamic_multimask_stability_thresh=0.98,  # Threshold for the stability score
        pred_obj_scores: bool = False,  # Whether to predict object scores 
        pred_obj_scores_mlp: bool = False,  # Whether to use an MLP for object score prediction
        use_multimask_token_for_obj_ptr: bool = False,  # Whether to use multimask output tokens for object pointers
        cfc_units: int = 128, 
        mixed_memory: bool = True, 
        cfc_mode: str = "default",
    ) -> None:
        """
        Initializes the MaskDecoder.

        Args:
            transformer_dim (int): Channel dimension of the transformer.
            transformer (layers.Layer): The transformer layer.
            num_multimask_outputs (int, optional): Number of masks to predict for 
                disambiguation. Defaults to 3.
            activation (Type[layers.Layer], optional): Activation function. 
                Defaults to layers.Activation('gelu').
            iou_head_depth (int, optional): Depth of the MLP for mask quality 
                prediction. Defaults to 3.
            iou_head_hidden_dim (int, optional): Hidden dimension of the MLP for 
                mask quality prediction. Defaults to 256.
            use_high_res_features (bool, optional): Use high-resolution features 
                in the decoder (if available from the backbone). Defaults to False.
            iou_prediction_use_sigmoid (bool, optional): Use sigmoid on the 
                output of IoU prediction head. Defaults to False.
            dynamic_multimask_via_stability (bool, optional): Dynamic multimask selection 
                based on stability score. Defaults to False.
            dynamic_multimask_stability_delta (float, optional): Delta used when computing the 
                stability score. Defaults to 0.05.
            dynamic_multimask_stability_thresh (float, optional): Threshold for the 
                stability score. Defaults to 0.98.
            pred_obj_scores (bool, optional): Whether to predict object scores 
                (in addition to masks). Defaults to False.
            pred_obj_scores_mlp (bool, optional): Whether to use an MLP for object 
                score prediction. Defaults to False.
            use_multimask_token_for_obj_ptr (bool, optional): Whether to use multimask 
                output tokens for object pointers. Defaults to False.
            cfc_units (int, optional): Number of units in the CfC layer. Defaults to 128.
            mixed_memory (bool, optional): Use mixed memory in CfC. Defaults to True.
            cfc_mode (str, optional): Mode for the CfC layer. Defaults to "default".
        """
        super(MaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = layers.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = layers.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = layers.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = tf.keras.Sequential([
            layers.Conv2DTranspose(
                filters=transformer_dim // 4, 
                kernel_size=2, 
                strides=2, 
                padding='same'
            ),
            LayerNorm2d(transformer_dim // 4),
            activation,
            layers.Conv2DTranspose(
                filters=transformer_dim // 8, 
                kernel_size=2, 
                strides=2, 
                padding='same' 
            ),
            activation,
        ])

        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = layers.Conv2D(transformer_dim // 8, kernel_size=1, strides=1, padding='same')
            self.conv_s1 = layers.Conv2D(transformer_dim // 4, kernel_size=1, strides=1, padding='same')

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        if self.pred_obj_scores:
            self.pred_obj_score_head = layers.Dense(1) 
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # --- CfC Layer Initialization ---
        self.cfc_layer = CfC(units=cfc_units, mixed_memory=mixed_memory, mode=cfc_mode) 

    def call(
        self,
        image_embeddings: tf.Tensor, # Image embeddings from the encoder
        image_pe: tf.Tensor, # Positional encodings for the image
        sparse_prompt_embeddings: tf.Tensor, # Sparse prompt embeddings
        dense_prompt_embeddings: tf.Tensor, # Dense prompt embeddings
        multimask_output: bool, # Whether to output multiple masks
        repeat_image: bool, # Whether to repeat the image embeddings
        high_res_features: Optional[List[tf.Tensor]] = None, # High-resolution features (optional)
        training=False, 
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: 
        """
        Predict masks given image and prompt embeddings.

        Returns:
            masks: Predicted masks
            iou_pred: Predicted IoU scores for the masks
            sam_tokens_out: SAM tokens for the mask output
            object_score_logits: Logits for object scores (if enabled)
        """

        # Predict masks using helper function
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            training=training 
        )

        # Select the appropriate masks based on output settings
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not training: 
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # Select SAM output tokens based on configuration
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:] 
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1] 

        # Return the predicted masks, IoU scores, SAM tokens, and object score logits
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: tf.Tensor, 
        image_pe: tf.Tensor, 
        sparse_prompt_embeddings: tf.Tensor, 
        dense_prompt_embeddings: tf.Tensor, 
        repeat_image: bool, 
        high_res_features: Optional[List[tf.Tensor]] = None, 
        training=False, 
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predicts masks. See 'call' for more details."""

        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = tf.concat(
                [
                    self.obj_score_token.weights[0],
                    self.iou_token.weights[0],
                    self.mask_tokens.weights[0],
                ],
                axis=0,
            )
            s = 1
        else:
            output_tokens = tf.concat(
                [self.iou_token.weights[0], self.mask_tokens.weights[0]], axis=0
            )
        output_tokens = tf.tile(tf.expand_dims(output_tokens, axis=0), [tf.shape(sparse_prompt_embeddings)[0], 1, 1])
        tokens = tf.concat((output_tokens, sparse_prompt_embeddings), axis=1)

        # Expand image embeddings if needed
        if repeat_image:
            src = tf.repeat(image_embeddings, repeats=tf.shape(tokens)[0], axis=0)
        else:
            tf.debugging.assert_equal(tf.shape(image_embeddings)[0], tf.shape(tokens)[0])
            src = image_embeddings
        src = src + dense_prompt_embeddings

        # Repeat image positional encodings
        tf.debugging.assert_equal(tf.shape(image_pe)[0], 1, message="image_pe should have size 1 in batch dim")
        pos_src = tf.repeat(image_pe, repeats=tf.shape(tokens)[0], axis=0)

        # Get shape of source tensor
        b, c, h, w = tf.shape(src)

        # --- Run the Transformer ---
        hs, src = self.transformer(src, pos_src, tokens, training=training)

        # --- Apply CfC Layer ---
        hs = self.cfc_layer(hs, training=training)  

        # Extract IoU token and mask tokens
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings
        src = tf.reshape(tf.transpose(src, perm=[0, 2, 1]), (b, c, h, w)) 
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            # Apply upscaling with high-resolution features
            dc1, ln1, act1, dc2, act2 = self.output_upscaling.layers
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Predict masks using hypernetworks
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = tf.stack(hyper_in_list, axis=1)
        b, c, h, w = tf.shape(upscaled_embedding)
        masks = tf.reshape(tf.linalg.matmul(hyper_in, tf.reshape(upscaled_embedding, (b, c, h * w))), (b, -1, h, w))

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Generate object score logits (if enabled)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * tf.ones_like(iou_pred[:, :1])

        # Return predicted masks, IoU predictions, mask tokens, and object score logits
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = tf.reshape(mask_logits, (tf.shape(mask_logits)[0], -1))
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = tf.math.reduce_sum(tf.cast(mask_logits > stability_delta, tf.float32), axis=-1)
        area_u = tf.math.reduce_sum(tf.cast(mask_logits > -stability_delta, tf.float32), axis=-1)
        stability_scores = tf.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = tf.math.argmax(multimask_iou_scores, axis=-1)
        batch_inds = tf.range(
            tf.shape(multimask_iou_scores)[0]
        )
        best_multimask_logits = tf.gather_nd(multimask_logits, tf.stack([batch_inds, best_scores_inds], axis=1))
        best_multimask_logits = tf.expand_dims(best_multimask_logits, axis=1)
        best_multimask_iou_scores = tf.gather_nd(multimask_iou_scores, tf.stack([batch_inds, best_scores_inds], axis=1))
        best_multimask_iou_scores = tf.expand_dims(best_multimask_iou_scores, axis=1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = tf.where(
            tf.broadcast_to(tf.expand_dims(is_stable, axis=-1), tf.shape(singlemask_logits)),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = tf.where(
            tf.broadcast_to(tf.expand_dims(is_stable, axis=-1), tf.shape(singlemask_iou_scores)),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out 
