# SAM 2 Export to ONNX and TFLITE

## Download model


```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## Requirements

onnx

```
torch 2.2.1
onnx 1.16.2
```

tflite

```
torch 2.4.0
ai-edge-torch 0.2.0
tf-nightly 2.18.0.dev20240905
```

## Export and Inference

onnx

```
python3 export_image_predictor.py --framework onnx
python3 export_video_predictor.py --framework onnx
```

tflite

```
export PJRT_DEVICE=CPU
python3 export_image_predictor.py --framework tflite
python3 export_video_predictor.py --framework tflite
```

## Inference only

onnx

```
python3 export_image_predictor.py --framework onnx --mode import
python3 export_video_predictor.py --framework onnx --mode import
```

tflite not supported inference only yet.

## Test

Replacing the complex tensor of RotaryEnc with matmul. To test this behavior, you can also run it with torch.

```
python3 export_video_predictor.py --framework torch
```

## Artifacts

The deliverables will be stored below.

```
output/*
model/*
```

You can also download it from the following.

### ONNX

- https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_t.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_t.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_t.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2/memory_encoder_hiera_t.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2/mlp_hiera_t.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_t.onnx (6dim matmul, batch = N)
- https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_t.opt.onnx (4dim matmul, batch = 1)

### TFLITE

- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/image_encoder_hiera_t.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/prompt_encoder_hiera_t.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/mask_decoder_hiera_t.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/mlp_hiera_t.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/memory_encoder_hiera_t.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2/memory_attention_hiera_t.tflite (4dim matmul, batch = 1, num_maskmem = 1)

The memory attention in tflite does not support dynamic shapes, so num_maskmem and max_obj_ptrs_in_encoder need to be fixed to 1.

## Inference Example

- [ailia-models](https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/segment-anything-2)
- [ailia-models-tflite](https://github.com/axinc-ai/ailia-models-tflite/pull/90)

## Original document

- [README_ORIGINAL.md](README_ORIGINAL.md)
