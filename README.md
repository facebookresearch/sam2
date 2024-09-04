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

```
output/*
model/*
```

## Inference Example

- [ailia-models](https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/segment-anything-2)
- [ailia-models-tflite](https://github.com/axinc-ai/ailia-models-tflite/pull/90)

## Original document

- [README_ORIGINAL.md](README_ORIGINAL.md)
