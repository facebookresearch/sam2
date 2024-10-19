# sam2keras/sam2_image_predictor.py

import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sam2.modeling.sam2_base import SAM2Base

class SAM2ImagePredictor(tf.keras.Model): 
    def __init__(self, model: SAM2Base):
        super(SAM2ImagePredictor, self).__init__() 
        self.model = model 

    def set_image(self, image: tf.Tensor, image_format: str = "RGB") -> None:
        """
        Set the image to be used for mask prediction.
        """
        # Image preprocessing (resize, normalize, convert to tensor)
        if image_format == "RGB":
            image = tf.image.convert_image_dtype(image, tf.float32)
        elif image_format == "BGR":
            image = tf.image.convert_image_dtype(image[:, :, ::-1], tf.float32) # BGR to RGB
        else:
            raise ValueError(f"Unsupported image format: {image_format}")

        input_image = tf.image.resize(image, (self.model.image_size, self.model.image_size))
        # Normalize (using same values as PyTorch version)
        input_image = (input_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        self.input_image = tf.expand_dims(input_image, axis=0)  

        # Extract backbone features
        self.features = self.model.forward_image(self.input_image)

        # Create placeholder output dictionary 
        self.output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        } 

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Predict masks for the given input prompts.
        """
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

        # High-resolution features for the SAM head
        if len(self.features["backbone_fpn"]) > 1:
            high_res_features = [
                tf.transpose(tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])), perm=[1, 0, 2])
                for x in self.features["backbone_fpn"][:-1]
            ]
        else:
            high_res_features = None

        # Predict masks 
        (
            low_res_masks,
            high_res_masks,
            ious,
            _,
            _,
            _,
            _,
        ) = self.model._forward_sam_heads(
            backbone_features=self.features["vision_features"], 
            high_res_features=high_res_features,
            multimask_output=multimask_output,
            **input_dict, # Pass other inputs 
        )
        
        # Return outputs 
        if return_logits:
            return low_res_masks, high_res_masks, ious
        else:
            masks = tf.math.sigmoid(high_res_masks) 
            return masks, ious, None # (No need to return 'transformed_point' in TF)
        
        '''
        Explanation and Adaptations:

Keras Model Subclassing: The SAM2ImagePredictor now directly subclasses tf.keras.Model.
set_image Method: Handles image preprocessing and feature extraction.
Assumes input image is a NumPy array and converts it to a TensorFlow tensor.
Applies resizing and normalization (using the same values as the PyTorch version).
Calls the forward_image method from the SAM2Base model to get the image features.
predict Method:
Takes input prompts as NumPy arrays and converts them to TensorFlow tensors.
Prepares a dictionary (input_dict) to pass prompts to the SAM heads.
Calls the _forward_sam_heads method from the SAM2Base model to predict masks.
Applies sigmoid to mask logits if return_logits is False.
Removal of transformed_point: The transformed_point output is not needed in the TensorFlow implementation.
TensorFlow/Keras Equivalents: Uses TensorFlow operations and functions like tf.convert_to_tensor, tf.expand_dims, tf.transpose, tf.reshape, tf.math.sigmoid, etc.
'''