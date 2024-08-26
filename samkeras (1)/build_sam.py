# sam2/build_sam.py

import tensorflow as tf
from tensorflow.keras import layers

from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam2_utils import MLP

def build_sam(
    image_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
):
    """
    Builds a SAM model.
    """

    # --- 1. Image Encoder (Using Hiera backbone and FPN) ---
    image_encoder = ImageEncoder(
        trunk=Hiera(
            embed_dim=embed_dim,
            num_heads=1,  # Initial number of heads
            drop_path_rate=0.0,  # Stochastic depth
            q_pool=3,  # Number of q_pool stages
            q_stride=(2, 2),  # Downsample stride between stages
            stages=(2, 3, 16, 3),  # Blocks per stage
            dim_mul=2.0,  # Dimension multiplier at stage shift
            head_mul=2.0,  # Head multiplier at stage shift
            window_pos_embed_bkg_spatial_size=(14, 14),
            window_spec=(8, 4, 14, 7),
            global_att_blocks=(12, 16, 20),
            return_interm_layers=True
        ),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=embed_dim, 
                feat_sizes=(image_size // patch_size, image_size // patch_size)
            ),
            d_model=embed_dim,
            backbone_channel_list=[96, 192, 384, 768], 
            kernel_size=1,
            stride=1,
            padding=0,
            fpn_interp_model="bilinear",
            fuse_type="sum",
            fpn_top_down_levels=None
        ),
        scalp=1  
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
    mask_decoder = MaskDecoder(
        transformer_dim=embed_dim,
        transformer=transformer,
        num_multimask_outputs=3,
        activation=layers.Activation('gelu'),
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=False,  # Set to True if using high-res features
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores=False,
        pred_obj_scores_mlp=False,
        use_multimask_token_for_obj_ptr=False,
    )

    # --- 4. Create the SAM Model ---
    
    # --- Inputs ---
    image_input = layers.Input(shape=(image_size, image_size, 3), name="image_input")
    point_coords_input = layers.Input(shape=(None, 2), dtype=tf.float32, name="point_coords")
    point_labels_input = layers.Input(shape=(None,), dtype=tf.int32, name="point_labels")
    # ... (Add box_input and/or mask_input if needed) 

    # --- Connect the Components ---
    image_embeddings = image_encoder(image_input)["vision_features"]
    sparse_embeddings, dense_embeddings = prompt_encoder(
        (point_coords_input, point_labels_input), 
        boxes=None,  # Connect if you have box_input
        masks=None,  # Connect if you have mask_input 
    )

    (
        low_res_masks,
        high_res_masks,
        ious,
        _,  # low_res_masks (duplicate)
        _,  # high_res_masks (duplicate)
        _,  # output_token
        object_score_logits, 
    ) = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,  # Set to True if you need multi-mask output
        repeat_image=False,
        high_res_features=None,  
    )

    # --- Outputs ---
    outputs = [low_res_masks, high_res_masks, ious, object_score_logits] 

    # --- Create the Model ---
    sam_model = tf.keras.Model(
        inputs=[image_input, point_coords_input, point_labels_input],  # Add other inputs if needed
        outputs=outputs 
    )
    return sam_model

'''
Explanation:

ImageEncoder (Hiera and FPN):
Uses the Hiera backbone and FpnNeck to create a multi-scale feature pyramid, as described in the SAM2 paper.
PromptEncoder:
Takes point coordinates and labels as inputs. Add box_input and/or mask_input if your SAM model uses those prompt types.
MaskDecoder:
Receives image embeddings from the ImageEncoder and prompt embeddings from the PromptEncoder.
Predicts masks, IoU scores, and object score logits.
Model Inputs and Outputs:
The tf.keras.Model is constructed to accept the necessary input prompts (image, points) and produce the desired outputs.

'''