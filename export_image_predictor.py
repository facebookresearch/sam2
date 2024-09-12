# Export image encoder and prompt encoder and mask decoder
# Implemented by ax Inc. 2024

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="hiera_t", choices=["hiera_l", "hiera_b+", "hiera_s", "hiera_t"])
parser.add_argument('--framework', default="onnx", choices=["onnx", "tflite", "torch"])
parser.add_argument('--accuracy', default="float", choices=["float", "int8"])
parser.add_argument('--mode', default="both", choices=["both", "import", "export"])
parser.add_argument('--image_size', default=1024, type=int, choices=[512, 1024])
args = parser.parse_args()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# output
os.makedirs("output", exist_ok=True)
os.makedirs("model", exist_ok=True)

# export settings
export_to_onnx_image_encoder = args.framework == "onnx" and (args.mode=="export" or args.mode=="both")
export_to_onnx_mask_decoder = args.framework == "onnx" and (args.mode=="export" or args.mode=="both")
import_from_onnx = args.framework == "onnx" and (args.mode=="import" or args.mode=="both")

export_to_tflite_image_encoder = args.framework == "tflite" and (args.mode=="export" or args.mode=="both")
export_to_tflite_mask_decoder = args.framework == "tflite" and (args.mode=="export" or args.mode=="both")
import_from_tflite = args.framework == "tflite" and (args.mode=="import" or args.mode=="both")

tflite_int8 = args.accuracy == "int8"

# export PJRT_DEVICE=CPU

# model settings
model_id = args.model_id
if model_id == "hiera_l":
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
elif model_id == "hiera_b+":
    sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
elif model_id == "hiera_s":
    sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
elif model_id == "hiera_t":
    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
else:
    print("unknown model id")
    exit()

# resolution settings
if args.image_size == 512:
    model_id = model_id + "_512"

# use cpu for export
device = torch.device("cpu")

# utility
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, model_id=model_id):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        #plt.show()
        plt.savefig(f'output/output{i+1}_'+model_id+'.png')

# logic
image = Image.open('notebooks/images/truck.jpg')
image = np.array(image.convert("RGB"))

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, image_size=args.image_size)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image, export_to_onnx = export_to_onnx_image_encoder,
                    export_to_tflite = export_to_tflite_image_encoder,
                    import_from_onnx = import_from_onnx, import_from_tflite = import_from_tflite,
                    tflite_int8 = tflite_int8, model_id = model_id)

input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    export_to_onnx=export_to_onnx_mask_decoder,
    export_to_tflite=export_to_tflite_mask_decoder,
    import_from_onnx=import_from_onnx,
    import_from_tflite=import_from_tflite,
    tflite_int8=tflite_int8,
    model_id=model_id
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, model_id=model_id)

print("Success!")