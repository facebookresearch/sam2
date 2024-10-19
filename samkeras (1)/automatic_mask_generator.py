# /automatic_mask_generator.py

import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

class AutomaticMaskGenerator(tf.keras.Model):
    def __init__(self, model: SAM2Base, points_per_side: Optional[int] = 32, points_per_batch: int = 64, pred_iou_thresh: float = 0.88, stability_score_thresh: float = 0.95, stability_score_offset: float = 1.0, box_nms_thresh: float = 0.7, crop_n_layers: int = 0, crop_nms_thresh: float = 0.7, mask_threshold: float = 0.0):
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of points prompts over the image, then filters
        low quality and duplicate masks.

        Arguments:
          model (SAM2Base): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit point
            sampling.
          points_per_batch (int): The number of points to be processed at once
            by the model. Larger values may be faster but use more memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift stability scores up
            or down when filtering.
          box_nms_thresh (float): The box IoU threshold for non-maximal
            suppression (NMS).
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer more crops than the previous.
          crop_nms_thresh (float): The box IoU threshold for NMS on crops.
          mask_threshold (float):  The threshold to use when binarizing the
            final mask predictions.
        """
        super(AutomaticMaskGenerator, self).__init__()
        self.model = model
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.mask_threshold = mask_threshold

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='image')
    ])
    def call(self, image: tf.Tensor) -> tf.Tensor:
        """
        Generates masks for the given image.

        Arguments:
          image (tf.Tensor): The image to generate masks for. Shape: [H, W, 3].

        Returns:
          tf.Tensor: A tensor of masks, shape [N, H, W] where N is the number
            of masks. 
        """
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        # Generate point prompts 
        if self.points_per_side is not None:
            point_grids = self._generate_point_grids(height, width)
            point_coords = tf.concat(point_grids, axis=0)
        else:
            raise NotImplementedError("Explicit point grids are not yet implemented in TensorFlow.")
        
        # Predict masks for each batch of points
        mask_predictions = []
        for i in range(0, tf.shape(point_coords)[0], self.points_per_batch):
            points_batch = point_coords[i: i + self.points_per_batch]
            masks, _, _ = self.predict_batch(image, points_batch) 
            mask_predictions.append(masks)
        
        masks = tf.concat(mask_predictions, axis=0) 

        # Filter masks based on predicted IoU and stability scores
        # (Implementation of filtering steps - adapt from PyTorch code) 
        ious = self._get_mask_ious(masks)
        keep_by_iou = tf.where(ious > self.pred_iou_thresh)[:, 0]
        masks = tf.gather(masks, keep_by_iou, axis=0) 
        ious = tf.gather(ious, keep_by_iou, axis=0)

        stability_scores = self._get_stability_scores(masks)
        keep_by_stability = tf.where(stability_scores > self.stability_score_thresh)[:, 0]
        masks = tf.gather(masks, keep_by_stability, axis=0)
        ious = tf.gather(ious, keep_by_stability, axis=0)

        keep_by_nms = tf.image.non_max_suppression(
            self._get_mask_bboxes(masks),
            ious,
            max_output_size=tf.shape(masks)[0], # Keep all for now
            iou_threshold=self.box_nms_thresh,
        )
        masks = tf.gather(masks, keep_by_nms, axis=0)

        # (Optional) Run on crops 
        if self.crop_n_layers > 0:
            crop_masks = self._generate_crop_masks(image, masks)
            masks = tf.concat([masks, crop_masks], axis=0)

            # NMS on combined masks (from points and crops)
            combined_ious = self._get_mask_ious(masks)
            keep_by_nms = tf.image.non_max_suppression(
                self._get_mask_bboxes(masks),
                combined_ious,
                max_output_size=tf.shape(masks)[0],
                iou_threshold=self.crop_nms_thresh, 
            )
            masks = tf.gather(masks, keep_by_nms, axis=0)

        return masks 
    
    def _generate_point_grids(self, height: int, width: int) -> List[tf.Tensor]:
        """Generates point grids for the given image dimensions."""
        if self.points_per_side is None:
            return []
        
        # Get the coordinates of the points 
        x = tf.linspace(0.0, tf.cast(width, tf.float32) - 1.0, self.points_per_side)
        y = tf.linspace(0.0, tf.cast(height, tf.float32) - 1.0, self.points_per_side) 
        xv, yv = tf.meshgrid(x, y) 
        point_coords = tf.stack([xv, yv], axis=-1)
        point_coords = tf.reshape(point_coords, (-1, 2)) # Flatten 

        # Split into batches
        point_grids = tf.split(point_coords, tf.shape(point_coords)[0] // self.points_per_batch, axis=0) 
        return point_grids

    def predict_batch(
        self, image: tf.Tensor, points: tf.Tensor, 
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: 
        """Predicts masks for a batch of points."""
        # Convert points to normalized coordinates
        points = points / tf.cast(tf.stack([tf.shape(image)[1], tf.shape(image)[0]]), dtype=tf.float32)

        # Create point labels (all positive)
        point_labels = tf.ones(tf.shape(points)[0], dtype=tf.int32)

        # Predict masks
        predictor = SAM2ImagePredictor(self.model)
        predictor.set_image(image)
        masks, ious, _ = predictor.predict(
            point_coords=tf.expand_dims(points, axis=0),
            point_labels=tf.expand_dims(point_labels, axis=0),
            multimask_output=True,
        )

        return masks[0], ious[0], None  # (No need for 'transformed_points' in TF)

    # --- Helper Functions for Filtering and Cropping ---

    def _get_mask_ious(self, masks: tf.Tensor) -> tf.Tensor:
        """
        Estimate the IoUs of the masks. 
        We use a fast approximation: The ratio of the area of the intersection
        to the area of the union is approximated by the ratio of the number of pixels
        in the intersection to the number of pixels in the union.
        """
        masks = tf.cast(masks > self.mask_threshold, tf.float32)
        intersection = tf.math.reduce_sum(masks, axis=0, keepdims=True)
        union = tf.math.reduce_sum(tf.cast(intersection > 0, tf.float32), axis=0)
        return tf.reshape(tf.where(union > 0, intersection / union, 0.0), (-1,))
    
    def _get_stability_scores(self, masks: tf.Tensor) -> tf.Tensor:
        """Computes the stability score of each mask."""
        # This implementation is adapted from the detectron2 version, but
        # it's not clear how well it works in this context.
        # You might need to experiment with other stability score functions.

        H, W = tf.shape(masks)[1], tf.shape(masks)[2]
        expanded_masks = tf.expand_dims(masks, axis=-1) 
        # Compute the area of each mask
        areas = tf.math.reduce_sum(tf.cast(expanded_masks > self.mask_threshold, tf.float32), axis=[1, 2]) 

        # Create a grid of x and y coordinates
        x = tf.tile(tf.expand_dims(tf.range(W, dtype=tf.float32), axis=0), [H, 1])
        y = tf.transpose(x)

        # Compute the center of mass for each mask
        x_center = tf.math.reduce_sum(x * tf.cast(expanded_masks > self.mask_threshold, tf.float32), axis=[1, 2]) / areas
        y_center = tf.math.reduce_sum(y * tf.cast(expanded_masks > self.mask_threshold, tf.float32), axis=[1, 2]) / areas

        # Threshold the masks at different levels
        delta = self.stability_score_offset 
        stability_scores = []
        for level in tf.linspace(self.mask_threshold - delta, self.mask_threshold + delta, 5):
            thresholded_masks = tf.cast(masks > level, tf.float32)
            intersection = tf.math.reduce_sum(thresholded_masks, axis=0, keepdims=True)
            union = tf.math.reduce_sum(tf.cast(intersection > 0, tf.float32), axis=0)
            iou = tf.where(union > 0, intersection / union, 0.0)
            stability_scores.append(tf.reshape(iou, (-1,)))
        stability_scores = tf.stack(stability_scores, axis=1) 

        # Compute the variance of the IoU values for each mask
        variance = tf.math.reduce_variance(stability_scores, axis=1)
        return 1.0 - variance
        
    def _get_mask_bboxes(self, masks: tf.Tensor) -> tf.Tensor:
        """Computes the bounding boxes of the masks."""
        h, w = tf.shape(masks)[1], tf.shape(masks)[2]
        rows = tf.tile(tf.expand_dims(tf.range(h), axis=1), [1, w])
        cols = tf.tile(tf.expand_dims(tf.range(w), axis=0), [h, 1])
        rows = tf.expand_dims(tf.reshape(rows, (1, -1)), axis=0)
        cols = tf.expand_dims(tf.reshape(cols, (1, -1)), axis=0)
        masks_reshaped = tf.reshape(masks, (tf.shape(masks)[0], -1))

        # Compute the bounding boxes
        y1 = tf.math.reduce_min(tf.boolean_mask(rows, tf.cast(masks_reshaped, tf.bool)), axis=1)
        x1 = tf.math.reduce_min(tf.boolean_mask(cols, tf.cast(masks_reshaped, tf.bool)), axis=1)
        y2 = tf.math.reduce_max(tf.boolean_mask(rows, tf.cast(masks_reshaped, tf.bool)), axis=1)
        x2 = tf.math.reduce_max(tf.boolean_mask(cols, tf.cast(masks_reshaped, tf.bool)), axis=1)
        bboxes = tf.stack([y1, x1, y2, x2], axis=1)
        return bboxes

    def _generate_crop_masks(self, image: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """Generates masks for crops of the image."""
        crop_masks = []
        for i in range(self.crop_n_layers):
            crop_size = tf.cast(tf.math.pow(2.0, i), tf.int32) 
            for y_start in range(0, tf.shape(image)[0], crop_size):
                for x_start in range(0, tf.shape(image)[1], crop_size):
                    # Crop the image
                    crop = tf.image.crop_to_bounding_box(
                        image, y_start, x_start, crop_size, crop_size
                    )

                    # Run the mask generator on the crop
                    crop_masks_batch = self.call(crop)
                    crop_masks.append(crop_masks_batch) 

        return tf.concat(crop_masks, axis=0) if crop_masks else tf.zeros((0, tf.shape(image)[0], tf.shape(image)[1]))
    
    
    '''Important Considerations:

Stability Score: The _get_stability_scores function in this code is adapted directly from the Detectron2 version. You may need to experiment and potentially find a more suitable stability score function for the SAM2 context.
Cropping Logic: This implementation performs cropping at multiple scales. Make sure this aligns with the behavior of the PyTorch version.
Computational Efficiency: The cropping logic can be computationally expensive. Consider optimizations or alternative methods if it becomes a performance bottleneck.'''