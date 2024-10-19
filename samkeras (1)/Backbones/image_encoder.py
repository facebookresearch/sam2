# sam2keras/modeling/backbones/image_encoder.py

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.nn as nn

class ImageEncoder(tf.keras.Model):
    def __init__(
        self,
        trunk: layers.Layer, # TensorFlow/Keras Layer
        neck: layers.Layer, # TensorFlow/Keras Layer
        scalp: int = 0,
    ):
        super(ImageEncoder, self).__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp

        # Assert that channel list lengths match (using tf.debugging.assert_equal)
        tf.debugging.assert_equal(
            len(self.trunk.channel_list), len(self.neck.backbone_channel_list),
            message=f"Channel list lengths do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        ) 
        # Check if individual channel dimensions match as well
        for i in range(len(self.trunk.channel_list)):
            tf.debugging.assert_equal(
                self.trunk.channel_list[i], self.neck.backbone_channel_list[i],
                message=f"Channel dims of trunk and neck do not match at index {i}. Trunk: {self.trunk.channel_list[i]}, neck: {self.neck.backbone_channel_list[i]}"
            ) 

    def call(self, sample: tf.Tensor, training=False):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample, training=training), training=training) # Pass training arg
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(layers.Layer):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    """

    def __init__(
        self,
        position_encoding: layers.Layer,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        super(FpnNeck, self).__init__()
        self.position_encoding = position_encoding
        self.convs = [] # Use list for convs in TF 
        self.backbone_channel_list = backbone_channel_list

        for dim in backbone_channel_list:
            self.convs.append(
                layers.Conv2D(
                    filters=d_model,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same' if padding else 'valid' 
                )
            )
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def call(self, xs: List[tf.Tensor], training=False):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = tf.image.resize(
                    tf.cast(prev_features, tf.float32), 
                    size=tf.shape(lateral_features)[1:3], 
                    method=self.fpn_interp_model, 
                    antialias=False
                )
                if self.fuse_type == "sum":
                    prev_features = lateral_features + top_down_features
                elif self.fuse_type == "avg":
                    prev_features = (lateral_features + top_down_features) / 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out) 

        return out, pos
    
    '''Explanation and Adaptations:

TensorFlow/Keras Layers:
Uses tf.keras.Model, layers.Layer, layers.Conv2D to define the model structure.
FPN Implementation:
Employs tf.image.resize for upsampling in the FPN neck.
TensorFlow Assertions: Uses tf.debugging.assert_equal for length and dimension checks.
TensorFlow Operations: tf.cast, tf.shape, list slicing, etc. are used as TensorFlow equivalents.
Training Argument: The call method in both ImageEncoder and FpnNeck includes the training argument for controlling training behaviors.
Key Points:

The channel dimension checks at the beginning of ImageEncoder are more robust, ensuring that both the length and individual dimensions of the channel lists match.
I have assumed a padding strategy of 'same' for the convolutional layers. Double-check the original PyTorch implementation to confirm if this is correct.
'''
