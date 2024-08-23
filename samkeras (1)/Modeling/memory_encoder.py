# sam2keras/modeling/memory_encoder.py

import tensorflow as tf
from tensorflow.keras import layers
from sam2keras.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d

class MaskDownSampler(layers.Layer):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=layers.Activation('gelu'), # Using Keras Activation layer
    ):
        super().__init__()
        num_layers = int(tf.math.log(float(total_stride)) / tf.math.log(float(stride)))
        assert stride**num_layers == total_stride
        self.encoder = tf.keras.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.add(
                layers.Conv2D(
                    filters=mask_out_chans,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same' if padding else 'valid' # Handle padding
                )
            )
            self.encoder.add(LayerNorm2d(mask_out_chans)) 
            self.encoder.add(activation)
            mask_in_chans = mask_out_chans

        self.encoder.add(layers.Conv2D(embed_dim, kernel_size=1, strides=1))

    def call(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(layers.Layer):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=1, # DepthwiseConv2D handles strides differently 
            padding='same' if padding else 'valid',
            depth_multiplier=1, # Use depth_multiplier for depthwise conv
            use_bias=False, # Following the ConvNeXt paper
        ) if use_dwconv else layers.Conv2D(
            filters=dim, 
            kernel_size=kernel_size, 
            padding='same' if padding else 'valid', 
            use_bias=False
        )
        self.norm = LayerNorm2d(dim, epsilon=1e-6)
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)
        
        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                shape=(dim,),
                initializer=tf.keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
                name='gamma'
            ) 
        else:
            self.gamma = None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else layers.Lambda(lambda x: x) 

    def call(self, x, training=False):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = tf.transpose(x, perm=[0, 2, 3, 1]) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x, training=training) 
        return x


class Fuser(layers.Layer):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = layers.Lambda(lambda x: x) # Identity in TF
        self.layers = [layer for _ in range(num_layers)]

        if input_projection:
            assert dim is not None
            self.proj = layers.Conv2D(dim, kernel_size=1, strides=1)

    def call(self, x, training=False):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, training=training) # Pass training argument 
        return x


class MemoryEncoder(layers.Layer):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = layers.Conv2D(in_dim, kernel_size=1, strides=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = layers.Lambda(lambda x: x) # Identity in TF
        if out_dim != in_dim:
            self.out_proj = layers.Conv2D(out_dim, kernel_size=1, strides=1)

    def call(
        self,
        pix_feat: tf.Tensor,
        masks: tf.Tensor,
        skip_mask_sigmoid: bool = False,
        training=False
    ):
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = tf.math.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x, training=training)  # Pass training argument 
        x = self.out_proj(x)

        pos = self.position_encoding(x) # .to(x.dtype) not needed in TF

        return {"vision_features": x, "vision_pos_enc": [pos]}
    
    
    '''
    Explanation:

Layer Structures: Uses tf.keras.Sequential to organize the layers in MaskDownSampler and Fuser.
ConvNeXt Adaptation: CXBlock is adapted using layers.DepthwiseConv2D and layers.Dense to implement the ConvNeXt block.
Layer Normalization and Activation: Employs LayerNorm2d (from utils.py) and layers.Activation for layer normalization and activation functions.
Drop Path: The DropPath class from sam2_utils.py will need to be implemented (refer to the PyTorch code).
TensorFlow Equivalents: Uses tf.math.sigmoid, tf.transpose, tf.reshape, and other TensorFlow equivalents for PyTorch operations.
Training Argument: The call method in CXBlock, Fuser, and MemoryEncoder includes the training argument for controlling training behavior.
Key Points:

I've made assumptions about the padding strategies based on common practices. You might need to adjust the padding arguments in the convolutional layers based on the original PyTorch implementation.
Remember that DropPath still requires a separate implementation in TensorFlow.
    '''