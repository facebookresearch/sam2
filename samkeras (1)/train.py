# /train.py

import tensorflow as tf
from .modeling.sam2_base import SAM2_Base
from .modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from .modeling.backbones.hieradet import Hiera
from .modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modeling.memory_encoder import MemoryEncoder, MaskDownSampler, CXBlock, Fuser
from .modeling.position_encoding import PositionEmbeddingSine
from .modeling.sam.transformer import RoPEAttention, Attention, TwoWayTransformer
from .modeling.sam.prompt_encoder import PromptEncoder
from .modeling.sam.mask_decoder import MaskDecoder
from .modeling.sam2_utils import LayerNorm2d, DropPath, MLP, get_1d_sine_pe, select_closest_cond_frames
import os
import cv2
import tarfile
import numpy as np
import subprocess
from tqdm import tqdm  # Import tqdm for progress bars

# --- 1. Configuration and Hyperparameters ---

# Data Parameters
data_path = '/path/to/your/SA-V/dataset'  # **Update this path!**
batch_size = 32
frames_per_sequence = 8
max_prompts_per_sequence = 2
image_size = 256

# Model Parameters
embed_dim = 256
backbone_stride = 16
memory_bank_size = 6

# SAM2 Hyperparameters (Example values - adjust based on the paper or your needs)
num_layers = 4
num_heads = 8
mlp_ratio = 4.0
# ... (Add other SAM2-specific hyperparameters from the config file)

# Training Parameters
learning_rate = 1e-4
epochs = 100

# --- 2. Create the Data Pipeline (with GStreamer and .tar file handling) ---

def decode_video_with_gstreamer(video_path):
    """Decode video using GStreamer."""
    command = [
        'gst-launch-1.0',
        'filesrc', 'location=' + video_path,
        '!', 'decodebin',
        '!', 'videoconvert',
        '!', 'videoscale',
        '!', 'appsink', 'emit-signals=true', 'sync=false', 'max-buffers=1', 'drop=true'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    frames = []
    while True:
        data = process.stdout.read()
        if not data:
            break
        # Extract Frame using OpenCV
        buffer = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        frames.append(frame)

    return frames

def decode_rle_masks(mask_data, image_height, image_width):
    """Decode RLE masks from the SA-V dataset."""
    masks = []
    for rle in mask_data:
        mask = np.zeros(image_height * image_width, dtype=np.uint8)
        j = 2
        for i in range(len(rle) // 2 - 1):
            start = rle[j]
            length = rle[j + 1]
            mask[start:start + length] = 1
            j += 2
        mask = mask.reshape(image_height, image_width)
        masks.append(mask)
    return masks

def load_and_preprocess_data(tar_info):
    """Loads and preprocesses data from a .tar file."""
    # Assuming video filename is 'video.mp4' and mask filename is 'masks.rle'
    # Adjust based on your SA-V dataset structure!
    video_path = [m.name for m in tar_info.members if m.name.endswith('.mp4')][0]
    mask_path = [m.name for m in tar_info.members if m.name.endswith('.rle')][0]

    # Extract video and mask data
    with tarfile.open(tar_info.name, 'r') as tar:
        video_data = tar.extractfile(video_path).read()
        mask_data = tar.extractfile(mask_path).read()

    # --- (Verify mask data format and decode) ---
    # Example: If masks are stored as a list of RLEs in a text file
    mask_data = mask_data.decode('utf-8').splitlines()
    mask_data = [[int(x) for x in line.strip().split()] for line in mask_data]

    # Decode video using GStreamer
    frames = decode_video_with_gstreamer(video_data)
    frames = [tf.convert_to_tensor(frame, dtype=tf.float32) for frame in frames]

    # Decode RLE masks (get height and width from the first mask)
    image_height, image_width = mask_data[0][:2]
    masks = decode_rle_masks(mask_data, image_height, image_width)
    masks = [tf.convert_to_tensor(mask, dtype=tf.float32) for mask in masks]

    # Resize and normalize
    frames = [tf.image.resize(frame, (image_size, image_size)) for frame in frames]
    frames = [(frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] for frame in frames]
    frames = tf.stack(frames, axis=0)

    masks = [tf.image.resize(mask, (image_size, image_size), method='nearest') for mask in masks]
    masks = tf.stack(masks, axis=0)

    # Generate prompts
    prompts = []
    for i in range(frames_per_sequence):
        prompt = generate_prompt(frames[i], masks[i])
        prompts.append(prompt)

    return frames, masks, prompts

def generate_prompt(frame, mask):
    """Generate a prompt (click, box, or mask) for a single frame."""
    prompt_type = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)

    if prompt_type == 0:  # Click
        indices = tf.where(mask)
        selected_index = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(indices)[0], dtype=tf.int32)
        coords = tf.gather(indices, selected_index)[:2]
        label = 1
        prompt = {'type': 'click', 'coords': coords, 'label': label}
    elif prompt_type == 1:  # Box
        y_indices, x_indices = tf.where(mask)
        y_min = tf.reduce_min(y_indices)
        x_min = tf.reduce_min(x_indices)
        y_max = tf.reduce_max(y_indices)
        x_max = tf.reduce_max(x_indices)
        box = [x_min, y_min, x_max, y_max]
        prompt = {'type': 'box', 'coords': box}
    else:  # Mask
        prompt = {'type': 'mask', 'mask': mask}
    return prompt

def data_generator(data_path, batch_size, frames_per_sequence, max_prompts_per_sequence, image_size):
    """Generates batches of training data."""
    tar_files = tf.io.gfile.glob(os.path.join(data_path, "*.tar"))
    dataset = tf.data.Dataset.from_tensor_slices(tar_files)
    dataset = dataset.interleave(
        lambda tar_file: tf.data.Dataset.from_generator(
            lambda: tarfile.open(tar_file.numpy().decode('utf-8'), 'r'), 
            output_types=tarfile.TarInfo,
            output_shapes=() 
        ),
        cycle_length=tf.data.AUTOTUNE,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda tar_info: tar_info.name.endswith('.mp4'))
    dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(data_path, batch_size, frames_per_sequence, max_prompts_per_sequence, image_size)

# --- 3. Build the SAM2 Model ---

# Image Encoder (with FPN)
image_encoder = ImageEncoder(
    trunk=Hiera(embed_dim=embed_dim, num_heads=num_heads, drop_path_rate=0.1, q_pool=3,
                 q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0, head_mul=2.0,
                 window_pos_embed_bkg_spatial_size=(14, 14),
                 window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20),
                 return_interm_layers=True),
    neck=FpnNeck(
        position_encoding=PositionEmbeddingSine(num_pos_feats=embed_dim, feat_sizes=(image_size // backbone_stride, image_size // backbone_stride)),
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

# Memory Attention
memory_attention_layer = MemoryAttentionLayer(
    activation='gelu',
    cross_attention=RoPEAttention(
        embedding_dim=embed_dim,
        num_heads=8,
        downsample_rate=2,
        dropout=0.0,
        kv_in_dim=None,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(32, 32),
    ),
    d_model=embed_dim,
    dim_feedforward=2048,
    dropout=0.1,
    pos_enc_at_attn=True,
    pos_enc_at_cross_attn_keys=True,
    pos_enc_at_cross_attn_queries=True,
    self_attention=Attention(
        embedding_dim=embed_dim,
        num_heads=8,
        downsample_rate=1,
        dropout=0.0,
        kv_in_dim=None
    )
)
memory_attention = MemoryAttention(
    d_model=embed_dim,
    pos_enc_at_input=True,
    layer=memory_attention_layer,
    num_layers=4,
    batch_first=True
)

# Memory Encoder
mask_downsampler = MaskDownSampler(
    embed_dim=embed_dim,
    kernel_size=4,
    stride=4,
    padding=0,
    total_stride=16,
    activation=layers.Activation('gelu')
)
fuser = Fuser(
    layer=CXBlock(dim=embed_dim, kernel_size=7, padding=3, drop_path=0.0, layer_scale_init_value=1e-6,
                  use_dwconv=True),
    num_layers=3,
    dim=None,
    input_projection=False
)
memory_encoder = MemoryEncoder(
    out_dim=embed_dim,
    mask_downsampler=mask_downsampler,
    fuser=fuser,
    position_encoding=PositionEmbeddingSine(
        num_pos_feats=embed_dim,
        feat_sizes=(image_size // backbone_stride, image_size // backbone_stride)
    ),
    in_dim=256
)

# Create the SAM2 Base Model
sam_model = SAM2Base(
    image_encoder=image_encoder,
    memory_attention=memory_attention,
    memory_encoder=memory_encoder,
    num_maskmem=memory_bank_size,
    image_size=image_size,
    backbone_stride=backbone_stride,
    sigmoid_scale_for_mem_enc=1.0,
    sigmoid_bias_for_mem_enc=0.0,
    binarize_mask_from_pts_for_mem_enc=False,
    use_mask_input_as_output_without_sam=False,
    max_cond_frames_in_attn=-1,
    directly_add_no_mem_embed=False,
    use_high_res_features_in_sam=False,
    multimask_output_in_sam=False,
    multimask_min_pt_num=1,
    multimask_max_pt_num=1,
    multimask_output_for_tracking=False,
    use_multimask_token_for_obj_ptr=False,
    iou_prediction_use_sigmoid=False,
    memory_temporal_stride_for_eval=1,
    add_all_frames_to_correct_as_cond=False,
    non_overlap_masks_for_mem_enc=False,
    use_obj_ptrs_in_encoder=False,
    max_obj_ptrs_in_encoder=16,
    add_tpos_enc_to_obj_ptrs=True,
    proj_tpos_enc_in_obj_ptrs=False,
    only_obj_ptrs_in_the_past_for_eval=False,
    pred_obj_scores=False,
    pred_obj_scores_mlp=False,
    fixed_no_obj_ptr=False,
    soft_no_obj_ptr=False,
    use_mlp_for_obj_ptr_proj=False,
    sam_mask_decoder_extra_args=None,
    compile_image_encoder=False
)

# --- 4. Define Loss Function, Metrics, and Optimizer ---

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss for binary classification."""
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1)) - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0))

def dice_loss(y_true, y_pred):
    """Dice loss for binary segmentation."""
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)  # Add 1 for numerical stability

def sam_loss_fn(ground_truth_masks, predicted_masks, ious, object_score_logits):
    """SAM loss function combining focal, dice, and IoU losses."""
    focal_loss_value = focal_loss(ground_truth_masks, predicted_masks)
    dice_loss_value = dice_loss(ground_truth_masks, predicted_masks)
    iou_loss_value = tf.reduce_mean(1.0 - ious)

    objectness_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(object_score_logits), object_score_logits
    )
    
    loss = (
        focal_loss_value + dice_loss_value + iou_loss_value + objectness_loss
    )
    return loss

# Recall Metrics
recall_at_50 = tf.keras.metrics.Recall(thresholds=0.5, name="recall_at_50")
recall_at_75 = tf.keras.metrics.Recall(thresholds=0.75, name="recall_at_75")
recall_at_90 = tf.keras.metrics.Recall(thresholds=0.9, name="recall_at_90")

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# --- 5. Training Loop ---
from tqdm import tqdm

@tf.function 
def train_step(images, masks, prompts):
    with tf.GradientTape() as tape:
        total_loss = 0.0
        for i in range(frames_per_sequence):
            frame = images[:, i, :, :, :]
            mask = masks[:, i, :, :, :]
            prompt = prompts[i]

            # Get predictions - assuming your model outputs: predicted_masks, ious, object_score_logits
            (
                predicted_masks,
                ious,
                _,
                _,
                _,
                _,
                object_score_logits,
            ) = sam_model._forward_sam_heads(
                backbone_features=sam_model.forward_image(tf.expand_dims(frame, axis=0))["vision_features"],
                point_inputs=prompt,
                multimask_output=sam_model.multimask_output_in_sam,
            )

            # Update recall metrics
            recall_at_50.update_state(mask, predicted_masks)
            recall_at_75.update_state(mask, predicted_masks)
            recall_at_90.update_state(mask, predicted_masks)

            # Calculate loss 
            loss = sam_loss_fn(mask, predicted_masks, ious, object_score_logits)
            total_loss += loss

        # Average loss over the sequence 
        loss = total_loss / frames_per_sequence

    # --- 2. Backpropagation ---
    gradients = tape.gradient(loss, sam_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, sam_model.trainable_variables))

    return loss, recall_at_50.result(), recall_at_75.result(), recall_at_90.result()

# --- Training ---
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    with tqdm(total=len(train_dataset), unit="batch") as pbar:
        for batch, (images, masks, prompts) in enumerate(train_dataset):
            loss, recall_50, recall_75, recall_90 = train_step(images, masks, prompts)
            pbar.set_description(
                f"Loss: {loss.numpy():.4f}, Recall@50: {recall_50.numpy():.4f}, Recall@75: {recall_75.numpy():.4f}, Recall@90: {recall_90.numpy():.4f}"
            )
            pbar.update(1)
    # Reset metrics at the end of each epoch
    recall_at_50.reset_states()
    recall_at_75.reset_states()
    recall_at_90.reset_states()
# --- Save the Trained Model ---
# ... (Save your model using sam_model.save() or other methods) 