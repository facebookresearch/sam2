import torch

# Check if CUDA is available and display the GPU information
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"CUDA is available. Using GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")

from PIL import Image
import numpy as np
import pandas as pd
from imutils.scopereader import MicroscopeDataReader
import dask.array as da
import sys
import os
import argparse
import logging
import tifffile as tiff
import cv2

def read_DLC_csv(csv_file_path):

    df = pd.read_csv(csv_file_path)

    #remove column names and set first row to new column name
    df.columns = df.iloc[0]
    df = df[1:]

    # Get the first row (which will become the second level of column names)
    second_level_names = df.iloc[0]

    # Create a MultiIndex for columns using the existing column names as the first level
    first_level_names = df.columns
    multi_index = pd.MultiIndex.from_arrays([first_level_names, second_level_names])

    # Set the new MultiIndex as the columns of the DataFrame
    df.columns = multi_index

    # Remove the first row from the DataFrame as it's now used for column names
    df = df.iloc[1:]

    # Removing the first column (index 0)
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index(drop=True)

    # Convert each column to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(isinstance(df.columns, pd.MultiIndex))
    print(list(df.columns))

    return df

def extract_coordinate_by_likelihood(df, bodyparts):
        # Step 1: Identify the 'likelihood' columns dynamically
    likelihood_cols = [col for col in df.columns if col[1] == 'likelihood']
    
    # Step 2: Compute the average likelihood per row without modifying the DataFrame
    avg_likelihood = df[likelihood_cols].mean(axis=1)
    
    # Step 3: Find the maximum average likelihood
    max_avg_likelihood = avg_likelihood.max()
    
    # Step 4: Identify the row(s) with the maximum average likelihood
    max_indices = avg_likelihood[avg_likelihood == max_avg_likelihood].index
    
    # Step 5: Randomly select one row if there is a tie
    if len(max_indices) > 1:
        selected_index = np.random.choice(max_indices)
    else:
        selected_index = max_indices[0]

    # Step 6: Retrieve the entire row of the selected entry
    selected_row = df.loc[[selected_index]]

    #extract x and y coordinates as list
    result = {}
    for bodypart in bodyparts:
        if bodypart in selected_row.columns.get_level_values(0):
            x_values = pd.to_numeric(selected_row[bodypart]['x'], errors='coerce')
            y_values = pd.to_numeric(selected_row[bodypart]['y'], errors='coerce')
            result[bodypart] = list(zip(x_values, y_values))

    return result

def create_frames_directory(video_path, max_frames=100):
    video_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(video_dir, 'frame_directory')
    os.makedirs(frames_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)
    
    frame_count = 0
    saved_count = 0
    while saved_count < max_frames:
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count % frame_step == 0:
            frame_filename = os.path.join(frames_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved frame {saved_count}/{max_frames}")
        
        frame_count += 1
    
    video.release()
    
    print(f"Extracted {saved_count} frames to {frames_dir} (from total {total_frames} frames)")
    return frames_dir

import numpy as np

def generate_masklet(predictor, video_path, coordinate, frame_number):
    print(f"Generating masklet for coordinate {coordinate} on frame {frame_number}")
    x, y = coordinate
    points = np.array([[x, y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_path)
    print("Initialized inference state")

    # Add points to model and get mask logits
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_number,
        obj_id=0,
        points=points,
        labels=labels,
    )
    
    print("Added point to the model")

    # Dictionary to store masks for video frames
    video_segments = {}
    print("Propagating masklet through video frames:")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {}
        
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)

            # Print the mask values (show unique values in mask)
            unique_values = np.unique(mask)
            print(f"Mask for frame {out_frame_idx}, object {out_obj_id} has unique values: {unique_values}")
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values of the binary mask
                binary_unique_values = np.unique(binary_mask)
                print(f"Binary mask for frame {out_frame_idx}, object {out_obj_id} has unique values: {binary_unique_values}")
                
                # Add the binary mask to video_segments
                video_segments[out_frame_idx][out_obj_id] = binary_mask
                print(f"Processed frame {out_frame_idx}, object {out_obj_id}")
            else:
                print(f"Warning: mask for frame {out_frame_idx} and object {out_obj_id} is not 2D. Skipping.")

    return video_segments


def save_masklet(frames_dir, video_segments):
    print("Saving binary masklet images")
    output_dir = os.path.join(frames_dir, 'output_binary')
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            
            # Check if the mask is empty or not in expected format
            if mask is None or mask.size == 0:
                print(f"Warning: mask for frame {frame_idx} and object {obj_id} is empty or None. Skipping.")
                continue

            # Ensure mask is in the correct format (uint8)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # Check mask dimensions (ensure it's 2D for binary mask)
            if len(mask.shape) == 2:  # Assuming it's a binary mask (grayscale)
                output_path = os.path.join(output_dir, f"binary_masklet_frame_{frame_idx:05d}_obj_{obj_id}.png")
                cv2.imwrite(output_path, mask)
            else:
                print(f"Warning: mask for frame {frame_idx} and object {obj_id} has an unexpected shape {mask.shape}. Skipping.")

        print(f"Saved binary masklet for frame {frame_idx}")
    
    print(f"Saved all binary masklet images to {output_dir}")
    
    
def main():
    
    print("Starting SAM2 Video Processing")
    
    SAM2_path = "/lisc/scratch/neurobiology/zimmer/schaar/code/github/segment-anything-2"

    # Define paths relative to the base path
    checkpoint = os.path.join(SAM2_path , "checkpoints", "sam2_hiera_large.pt")
    model_cfg = "/" + os.path.join(SAM2_path, "sam2_configs", "sam2_hiera_l.yaml")

    # Check if files exist
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Config file not found: {model_cfg}")
    
    coordinate = (74, 96)
    frame_number = 10

    video_path = "/lisc/scratch/neurobiology/zimmer/schaar/Behavior/High_Res_Population/110620024/test_SAM2/2024-06-10_14-58-26_trainingsdata_clean3/2024-06-10_14-58-26_trainingsdata_clean3_track_0/output/track.avi"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Extracting frames from video")
    frames_dir = create_frames_directory(video_path, max_frames=100)

    print("Loading SAM2 model")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    print("Generating masklet")
    video_segments = generate_masklet(predictor, frames_dir, coordinate, frame_number)
    
    print("Masklet generated across the video.")
    
    print("Saving masklet images")
    save_masklet(frames_dir, video_segments)
    
    print("Processing complete")
    return video_segments

if __name__ == "__main__":
    main()