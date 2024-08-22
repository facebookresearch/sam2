# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

import onnxruntime

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
        export_to_onnx = False,
        export_to_tflite = False,
        import_from_onnx = False,
        import_from_tflite = False,
        model_id=None
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")

        if export_to_onnx:
            print("input_image", input_image.shape)
            torch.onnx.export(
                self.model, (input_image), 'image_encoder_'+model_id+'.onnx',
                input_names=["input_image"],
                output_names=["feats1", "feats2", "feats3"],
                verbose=False, opset_version=17
            )
        
        if import_from_onnx:
            model = onnxruntime.InferenceSession("image_encoder_"+model_id+".onnx")
            vision_feat1, vision_feat2, vision_feat3 = model.run(None, {"input_image":input_image.numpy()})
            feats = [torch.Tensor(vision_feat1), torch.Tensor(vision_feat2), torch.Tensor(vision_feat3)]
            #print("feats", vision_feat1.shape, vision_feat2.shape, vision_feat3.shape)

        if export_to_tflite:
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (input_image,)

            export_float = True
            export_int8 = False

            if export_float:
                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]}}
                edge_model = ai_edge_torch.convert(self.model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("image_encoder_"+model_id+".tflite")
                if import_from_tflite:
                    vision_feat1, vision_feat2, vision_feat3 = edge_model(sample_inputs)
                    feats = [torch.Tensor(vision_feat1), torch.Tensor(vision_feat2), torch.Tensor(vision_feat3)]

            if export_int8:
                from ai_edge_torch.quantize import pt2e_quantizer
                from ai_edge_torch.quantize import quant_config
                from torch.ao.quantization import quantize_pt2e

                quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
                    pt2e_quantizer.get_symmetric_quantization_config()
                )
                model = torch._export.capture_pre_autograd_graph(self.model, sample_inputs)
                model = quantize_pt2e.prepare_pt2e(model, quantizer)
                #model(input_image.type(torch.FloatTensor)) # calibration           
                model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]}}
                with_quantizer = ai_edge_torch.convert(
                    model,
                    sample_inputs,
                    quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    _ai_edge_converter_flags=tfl_converter_flags
                )
                with_quantizer.export("image_encoder_int8_"+model_id+".tflite")

                if import_from_tflite:
                    vision_feat1, vision_feat2, vision_feat3 = model(sample_inputs)
                    feats = [torch.Tensor(vision_feat1), torch.Tensor(vision_feat2), torch.Tensor(vision_feat3)]

        if not import_from_onnx and not import_from_tflite:
            backbone_out = self.model.forward_image(input_image)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        export_to_onnx=False,
        export_to_tflite=False,
        import_from_onnx = False,
        import_from_tflite = False,
        model_id=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            export_to_onnx=export_to_onnx,
            export_to_tflite=export_to_tflite,
            import_from_onnx=import_from_onnx,
            import_from_tflite=import_from_tflite,
            model_id=model_id
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
        export_to_onnx = False,
        export_to_tflite = False,
        import_from_onnx = False,
        import_from_tflite = False,
        model_id = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)


        if export_to_onnx:
            #print("concat_points", concat_points.shape)
            #print("mask_input", mask_input.shape)
            self.model.sam_prompt_encoder.forward = self.model.sam_prompt_encoder.forward_sparse
            torch.onnx.export(
                self.model.sam_prompt_encoder, (concat_points[0], concat_points[1]), 'prompt_encoder_sparse_'+model_id+'.onnx',
                input_names=["coords", "labels"],
                output_names=["sparse_embeddings", "dense_embeddings"],
                dynamic_axes={
                    'coords': {0: 'b', 1: 'n'},
                    'labels': {0: 'b', 1: 'n'},
                },
                verbose=False, opset_version=17
            )

        if import_from_onnx:
            model = onnxruntime.InferenceSession("prompt_encoder_sparse_"+model_id+".onnx")
            sparse_embeddings, dense_embeddings = model.run(None, {"coords":concat_points[0].numpy(), "labels":concat_points[1].numpy()})
            sparse_embeddings = torch.Tensor(sparse_embeddings)
            dense_embeddings = torch.Tensor(dense_embeddings)

        #if export_to_onnx:
            #self.model.sam_prompt_encoder.forward = self.model.sam_prompt_encoder.forward_dense
            #if mask_input is None:
            #    mask_input_non_zero = np.zeros((1, 1024, 1024))
            #else:
            #    mask_input_non_zero = mask_input
            #torch.onnx.export(
            #    self.model.sam_prompt_encoder, (mask_input_non_zero), 'prompt_encoder_dense_'+model_id+'.onnx',
            #    input_names=["mask_input"],
            #    output_names=["sparse_embeddings", "dense_embeddings"],
            #    verbose=False, opset_version=17
            #)

        if export_to_tflite:
            import ai_edge_torch
            sample_inputs = (concat_points[0], concat_points[1])
            self.model.sam_prompt_encoder.forward = self.model.sam_prompt_encoder.forward_sparse
            edge_model = ai_edge_torch.convert(self.model.sam_prompt_encoder, sample_inputs)
            edge_model.export("prompt_encoder_sparse_"+model_id+".tflite")

        if not import_from_onnx and not import_from_tflite:
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder.forward_normal(
                coords=concat_points[0],
                labels=concat_points[1],
                #boxes=None,
                masks=mask_input,
            )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        #print("sparse_embeddings", sparse_embeddings.shape)
        #print("dense_embeddings", dense_embeddings.shape)

        if export_to_onnx:
            torch.onnx.export(
                self.model.sam_mask_decoder, (self._features["image_embed"][img_idx].unsqueeze(0), self.model.sam_prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings, multimask_output, batched_mode, high_res_features[0], high_res_features[1]),
                'mask_decoder_'+model_id+'.onnx',
                input_names=["image_embeddings", "image_pe", "sparse_prompt_embeddings", "dense_prompt_embeddings", "multimask_output", "repeat_image", "high_res_features1", "high_res_features2"],
                output_names=["low_res_masks", "iou_predictions"],
                verbose=False, opset_version=17
            )
        
        if import_from_onnx:
            model = onnxruntime.InferenceSession("mask_decoder_"+model_id+".onnx")
            low_res_masks, iou_predictions, _, _  = model.run(None, {
                "image_embeddings":self._features["image_embed"][img_idx].unsqueeze(0).numpy(),
                "image_pe": self.model.sam_prompt_encoder.get_dense_pe().numpy(),
                "sparse_prompt_embeddings": sparse_embeddings.numpy(),
                "dense_prompt_embeddings": dense_embeddings.numpy(),
                "high_res_features1":high_res_features[0].numpy(),
                "high_res_features2":high_res_features[1].numpy()})
            low_res_masks = torch.Tensor(low_res_masks)
            iou_predictions = torch.Tensor(iou_predictions)

        if export_to_tflite:
            import ai_edge_torch
            sample_inputs = (self._features["image_embed"][img_idx].unsqueeze(0), self.model.sam_prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings, multimask_output, batched_mode, high_res_features[0], high_res_features[1])
            edge_model = ai_edge_torch.convert(self.model.sam_mask_decoder, sample_inputs)
            edge_model.export("mask_decoder_"+model_id+".tflite")

        if not import_from_onnx and not import_from_tflite:
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features1=high_res_features[0],
                high_res_features2=high_res_features[1],
            )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
