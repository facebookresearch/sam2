Usage:

You don't have to fork or copy anything!

1) Just grab the Snake Rule and implement it in your Snake Pipeline
2) Use the provided cluster_config rule to properly assign GPU ressources on the cluster

On Server:

Files and Script are located in shared Folder:

/lisc/scratch/neurobiology/zimmer/autoscope/code/segment-anything-2


SAM2 Video Processing Script

This script processes a video file to generate segmentation masks using DeepLabCut (DLC) outputs and the SAM2 (Segment Anything Model) for video object segmentation.

Overview

Input: Video file and corresponding DLC CSV file containing body part coordinates.
Process:
- Reads DLC CSV to extract the most confident frame and body part coordinates.
- Optionally adjusts coordinates if the DLC was performed on a downsampled video.
- Splits the video into batches of frames.
- Uses SAM2 to generate segmentation masks based on the provided coordinates.
- Propagates the masks forward and backward through the video frames.
Output: A single BTF Stack containing the segmentation masks for each frame.

Prerequisites

- Python 3.12
- Required Python packages:
  - torch
  - numpy
  - pandas
  - opencv-python
  - tifffile
  - Pillow
  - argparse
  - shutil
- SAM2 model and configuration files





