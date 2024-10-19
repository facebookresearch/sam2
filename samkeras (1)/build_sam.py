# sam2/build_sam.py
# sam2_tfkeras/build_sam.py

import tensorflow as tf
from sam2_tfkeras.modeling.sam.transformer import TwoWayTransformer
from sam2_tfkeras.modeling.sam.mask_decoder import MaskDecoder
from sam2_tfkeras.modeling.sam2_utils import MLP
from sam2_tfkeras.modeling.sam.prompt_encoder import PromptEncoder
from sam2_tfkeras.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2_tfkeras.modeling.backbones.hieradet import Hiera
from sam2_tfkeras.modeling.position_encoding import PositionEmbeddingSine
from ncps.wirings import AutoNCP

def build_sam(
    image_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    cfc_units: int = 128,  # Number of units in the CfC layer
    mixed_memory: bool = True,  # Use mixed memory in CfC
    wiring_type: str = 'full',  # 'full' or 'ncp'
    # ... (Other SAM-specific hyperparameters)
):
    """
    Builds a SAM model, incorporating a CfC layer and optional structured wiring.

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of the image patches.
        embed_dim (int): The embedding dimension.
        depth (int): The depth of the transformer.
        num_heads (int): The number of attention heads.
        mlp_ratio (float): The MLP expansion ratio.
        cfc_units (int): Number of units in the CfC layer.
        mixed_memory (bool): Whether to use mixed memory in CfC.
        wiring_type (str): The type of wiring for the CfC layer ('full' or 'ncp').
        # ... (Other hyperparameters) 

    Returns:
        tf.keras.Model: The SAM model.
    """

    # --- 1. Image Encoder (Hiera) ---
    image_encoder = ImageEncoder(
        trunk=Hiera(
            embed_dim=96,  
            num_heads=1,  
            drop_path_rate=0.0,  
            q_pool=3, 
            q_stride=(2, 2),  
            stages=(2, 3, 16, 3),  
            dim_mul=2.0,  
            head_mul=2.0,  
            window_pos_embed_bkg_spatial_size=(14, 14),
            window_spec=(8, 4, 14, 7),
            global_att_blocks=(12, 16, 20,),
            return_interm_layers=True,  
        ),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=embed_dim,
                feat_sizes=(image_size // patch_size, image_size // patch_size)
            ),
            d_model=embed_dim,
            backbone_channel_list=[96, 192, 384, 768], # Adjust if needed
            kernel_size=1,
            stride=1,
            padding=0,
            fpn_interp_model="bilinear",
            fuse_type="sum",
            fpn_top_down_levels=None, 
        )
    )

    # --- 2. Prompt Encoder --- 
    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_size // patch_size, image_size // patch_size), 
        input_image_size=(image_size, image_size),
        mask_in_chans=16, 
    )

    # --- 3. Mask Decoder ---
    transformer = TwoWayTransformer(
        depth=depth,
        embedding_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=int(embed_dim * mlp_ratio),
    )

    # --- Determine Wiring Type for the CfC Layer ---
    if wiring_type == 'ncp':
        wiring = AutoNCP(cfc_units, 3) # 3 outputs (masks, ious, etc.) - adjust if needed
    elif wiring_type == 'full':
        wiring = cfc_units # Use fully connected
    else:
        raise ValueError(f"Invalid wiring_type: {wiring_type}")

    mask_decoder = MaskDecoder(
        transformer_dim=embed_dim,
        transformer=transformer,
        num_multimask_outputs=3,
        cfc_units=cfc_units, # Pass the number of CfC units
        mixed_memory=mixed_memory, 
        # ... (Other mask decoder parameters)
    )

    # --- 4. Create the SAM Model ---
    # Define input layers
    image_input = layers.Input(shape=(image_size, image_size, 3), name='image_input') 
    point_coords_input = layers.Input(shape=(None, 2), dtype=tf.float32, name='point_coords')
    point_labels_input = layers.Input(shape=(None,), dtype=tf.int32, name='point_labels')
    box_input = layers.Input(shape=(4,), dtype=tf.float32, name='box')
    mask_input = layers.Input(shape=(image_size, image_size, 1), dtype=tf.float32, name='mask_input')

    # Pass inputs through the encoder and prompt encoder
    image_embeddings = image_encoder(image_input) 
    sparse_embeddings, dense_embeddings = prompt_encoder(
        (point_coords_input, point_labels_input), box_input, mask_input
    )

    # Get outputs from the mask decoder
    outputs = mask_decoder(
        image_embeddings=image_embeddings['vision_features'],
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,  # Set this based on your requirements
        repeat_image=False, 
    )

    # Create the Keras model 
    sam_model = tf.keras.Model(
        inputs=[image_input, point_coords_input, point_labels_input, box_input, mask_input], 
        outputs=outputs 
    ) 
    return sam_model

