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
python3 export_image_predictor.py --framework tflite
python3 export_video_predictor.py --framework tflite
```

## Inference only

onnx

```
python3 export_image_predictor.py --framework onnx --mode import
python3 export_video_predictor.py --framework onnx --mode import
```

tflite

```
python3 export_image_predictor.py --framework tflite --mode import
python3 export_video_predictor.py --framework tflite --mode import
```

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

## Original document

[README_ORIGINAL.md](README_ORIGINAL.md)
