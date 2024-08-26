# sam2keras/sam2_video_predictor.py

import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sam2.modeling.sam2_base import SAM2Base

class SAM2VideoPredictor(tf.keras.Model):
    def __init__(self, model: SAM2Base):
        super(SAM2VideoPredictor, self).__init__()
        self.model = model
        self.reset_state()

    def reset_state(self):
        """Reset the model's state, clearing all saved frames and masks."""
        self.frames = []
        self.features = []
        self.is_image_set = False

        # Placeholders for the output dictionary, to be filled in 'predict'
        self.output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

    def set_video(self, video: tf.Tensor, image_format: str = "RGB") -> None:
        """
        Set the video to be used for mask prediction.
        """
        if image_format == "RGB":
            video = tf.image.convert_image_dtype(video, tf.float32)
        elif image_format == "BGR":
            video = tf.image.convert_image_dtype(video[:, :, :, ::-1], tf.float32)  # BGR to RGB
        else:
            raise ValueError(f"Unsupported image format: {image_format}")

        # Resize and normalize each frame 
        self.frames = [tf.image.resize(frame, (self.model.image_size, self.model.image_size)) for frame in video]
        self.frames = [(frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] for frame in self.frames]

        # Extract backbone features for each frame
        self.features = [self.model.forward_image(tf.expand_dims(frame, axis=0)) for frame in self.frames]

        self.is_image_set = True

    def predict(
        self,
        frame_idx: int,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Predict masks for the given frame, using the provided prompts.
        """
        if not self.is_image_set:
            raise RuntimeError("An image or video must be set with .set_image or .set_video before calling .predict")

        if point_coords is not None:
            point_coords = tf.convert_to_tensor(point_coords, dtype=tf.float32)
        if point_labels is not None:
            point_labels = tf.convert_to_tensor(point_labels, dtype=tf.int32)
        if box is not None:
            box = tf.convert_to_tensor(box, dtype=tf.float32)
        if mask_input is not None:
            mask_input = tf.convert_to_tensor(mask_input, dtype=tf.float32)
            # Add batch dimension if missing
            if len(mask_input.shape) == 3:
                mask_input = tf.expand_dims(mask_input, axis=0)

        # Prepare inputs
        input_dict = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box": box,
            "mask_input": mask_input,
        }

        # Get features and sizes
        backbone_out = self.features[frame_idx]
        (
            backbone_out,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.model._prepare_backbone_features(backbone_out)

        # Determine if this is an initial conditioning frame
        is_init_cond_frame = frame_idx in self.output_dict["cond_frame_outputs"]

        # Track the object through the video
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            output_dict=self.output_dict,
            num_frames=len(self.frames),
            **input_dict, # Pass other inputs 
        )

        # Save frame outputs
        if is_init_cond_frame:
            self.output_dict["cond_frame_outputs"][frame_idx] = current_out
        else:
            self.output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        # Return outputs
        if return_logits:
            return current_out["pred_masks"], current_out["pred_masks_high_res"], None 
        else:
            masks = tf.math.sigmoid(current_out["pred_masks_high_res"])
            return masks, None, None 
        
        
        '''
        Explanation and Adaptations:

Keras Model: The SAM2VideoPredictor now directly subclasses tf.keras.Model.
reset_state Method: Resets internal state (frames, features, output dictionary) to allow for processing a new video.
set_video Method:
Takes a video (assumed to be a TensorFlow tensor) and preprocesses each frame (resizing, normalization).
Extracts backbone features for each frame using model.forward_image.
predict Method:
Takes the frame index and prompts as input.
Gets the features for the specified frame.
Determines if it's an initial conditioning frame.
Calls model.track_step to perform the tracking for that frame.
Saves the frame outputs to the output_dict.
Applies sigmoid to the mask logits if return_logits is False.
TensorFlow/Keras Operations: Uses TensorFlow functions like tf.convert_to_tensor, tf.expand_dims, tf.math.sigmoid, etc.
Key Points:

This implementation follows common TensorFlow/Keras conventions and integrates well with the other converted modules.
The logic closely follows the original PyTorch version, but it's adapted for TensorFlow tensors and operations.
The reset_state method allows for efficient processing of multiple videos.'''