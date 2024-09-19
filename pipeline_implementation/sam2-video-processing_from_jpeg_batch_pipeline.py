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
    
    return result, selected_index

def create_frames_directory(video_path, batch_size):
    video_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(video_dir, 'frame_directory')
    os.makedirs(frames_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"No frames found in video file: {video_path}")
    
    saved_count = 0
    batch_number = 0
    batch_frame_counts = [] #saves batchnumber and frames inside this batchnumber
    current_batch_count = 0
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Create a new batch subfolder when necessary
            if current_batch_count == 0:
                batch_dir = os.path.join(frames_dir, f"{batch_number}")
                os.makedirs(batch_dir, exist_ok=True)
            
            frame_filename = os.path.join(batch_dir, f"{saved_count:06d}.jpg")  # 6 digits for larger video frame counts
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            current_batch_count += 1
            
            if current_batch_count == batch_size:  # Close batch when full
                batch_frame_counts.append((batch_number, current_batch_count))
                current_batch_count = 0
                batch_number += 1
            
            print(f"Saved frame {saved_count}/{total_frames} in folder {batch_number}")
        
        # Add the last batch to the list if it wasn't full
        if current_batch_count > 0:
            batch_frame_counts.append((batch_number, current_batch_count))
    
    finally:
        video.release()
    
    print(f"Extracted {saved_count} frames to {frames_dir} (from total {total_frames} frames)")
    print(f"Organized into {batch_number + 1} folders")
    
    return frames_dir, batch_frame_counts

def segment_object(predictor, video_path, coordinate, frame_number, batch_number, batch_size):

    print("Batch Folder Path", video_path)

    print(f"Generating masklet for coordinate {coordinate} on frame {frame_number}")

    #model only sees 1 batch as whole video
    normalized_frame_number = frame_number - (batch_number * batch_size)
    print("Normalized Framenumber for Batch:", normalized_frame_number)

     # Initialize lists to hold points and labels
    points_list = []
    labels_list = []
    
    # Iterate over each key (e.g., 'vulva', 'neck') in the coordinate dictionary
    for key in coordinate:
        # Iterate over each (x, y) tuple in the list associated with the key
        for (x, y) in coordinate[key]:
            points_list.append([x, y])
            labels_list.append(1)  # You can customize the label as needed

    # Convert the lists to NumPy arrays
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Now you can use 'points' and 'labels' as needed
    print("Points array:", points)
    print("Labels array:", labels)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_path)
    print("Initialized inference state")

    # Add points to model and get mask logits
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=normalized_frame_number,
        obj_id=0,
        points=points,
        labels=labels,
    )
    
    print("Added point to the model")

    # Dictionary to store masks for video frames
    video_segments = {}
    print("Propagating masklet through video frames (forward direction):")

    # Forward propagation (reverse=False)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key, just out_frame_idx)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Reverse propagation (reverse=True)
    print("Propagating masklet through video frames (reverse direction):")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Sort the dictionary to ensure the frames are in proper order
    video_segments = dict(sorted(video_segments.items()))

    return video_segments

def process_mask(mask):
    """
    Process the input mask to ensure it's in the proper format:
    8-bit, single-channel binary image.
    
    Args:
    mask (numpy.ndarray): Input mask image
    
    Returns:
    numpy.ndarray: Processed binary mask
    """
    # Check the number of channels
    if len(mask.shape) == 2:
        # If it's already single channel
        mask_single_channel = mask
    elif len(mask.shape) == 3:
        if mask.shape[2] == 3:
            # If it's a 3-channel image, convert to grayscale
            mask_single_channel = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            # If it's a 4-channel image (with alpha), convert to grayscale
            mask_single_channel = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unexpected number of channels: {mask.shape[2]}")
    else:
        raise ValueError(f"Unexpected shape of mask: {mask.shape}")

    # Ensure 8-bit depth
    mask_8bit = cv2.convertScaleAbs(mask_single_channel)

    # Apply threshold to create binary mask
    _, mask_binary = cv2.threshold(mask_8bit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return mask_binary

    
    
def main(args):

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process input files and generate output.")
    parser.add_argument("-video_path", type=str, help="Path to the input file")
    parser.add_argument("-output_file_path", type=str, help="Path to the output file")
    parser.add_argument("-DLC_csv_file_path", type=str, help="Path to the DLC CSV file")
    parser.add_argument("-column_names", type=str, nargs='+', help="List of column names")
    parser.add_argument("-SAM2_path", type=str, help="Location of GitRepo!")
    parser.add_argument("--downsample_factor", type=float, default=0, help="Use in case DLC was used on downsampled video (default: 0)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing (default: 100)")

    # Parse the arguments
    args = parser.parse_args(args)
    
    print("Starting SAM2 Video Processing")
    
    SAM2_path = args.SAM2_path

    # Define paths relative to the base path
    checkpoint = os.path.join(SAM2_path , "checkpoints", "sam2_hiera_large.pt")
    model_cfg = "/" + os.path.join(SAM2_path, "sam2_configs", "sam2_hiera_l.yaml")

    # Check if files exist
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Config file not found: {model_cfg}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from sam2.build_sam import build_sam2_video_predictor

    print("Loading SAM2 model")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    # Access the parsed arguments
    video_path = args.video_path
    output_file_path = args.output_file_path
    DLC_csv_file_path = args.DLC_csv_file_path
    column_names = args.column_names
    downsample_factor = args.downsample_factor
    batch_size = args.batch_size

    print(f"Video File: {video_path}")
    print(f"Output file: {output_file_path}")
    print(f"DLC CSV file: {DLC_csv_file_path}")
    print(f"Column names: {column_names}")
    print(f"Batchsize: {batch_size}")

    print("Extracting frames from video")
    frames_dir, batch_frame_count = create_frames_directory(video_path, batch_size)

    DLC_data = read_DLC_csv(DLC_csv_file_path)

    # Initialize a variable to hold the final concatenated mask
    final_mask_dict = {}

    for batch_number, _ in batch_frame_count:
        # Directory for the current batch of frames
        batch_dir = os.path.join(frames_dir, f"{batch_number}")

        # Slice the DataFrame for the current batch
        DLC_data_batch = DLC_data.iloc[batch_number * batch_size : (batch_number + 1) * batch_size]

        # Extract coordinates and frame numbers for the current batch
        coordinates, frame_number = extract_coordinate_by_likelihood(DLC_data_batch, column_names)

        # Generate mask for the current batch of frames
        print(f"Generating masklet for batch {batch_number}")

        masks = segment_object(predictor, batch_dir, coordinates, frame_number, batch_number, batch_size)
        print(f"Masklet generated for batch {batch_number}. Number of masks: {len(masks)}. Stored masks: {list(masks.keys())}")

         # Concatenate masks into the final dictionary with global frame indexing
        for out_frame_idx, binary_mask in masks.items():
            global_frame_idx = out_frame_idx + (batch_number * batch_size)  # Adjust frame index to global frame count
            final_mask_dict[global_frame_idx] = binary_mask  # Store or update the mask for the global frame

    """
    Process and save the final mask dictionary into a TIFF file.
    """
    # Sort the dictionary by the global frame index (just in case)
    sorted_mask_keys = sorted(final_mask_dict.keys())

    with tiff.TiffWriter(output_file_path, bigtiff=True) as tif_writer:
        for global_frame_idx in sorted_mask_keys:
            mask = final_mask_dict[global_frame_idx]
            
            # Process the mask (e.g., convert to binary, 8-bit)
            processed_mask = process_mask(mask)
            print(f"Saving mask for global frame {global_frame_idx} with shape: {processed_mask.shape}")

            # Ensure the mask is not empty before saving
            if np.any(processed_mask):
                tif_writer.write(processed_mask, contiguous=True)
            else:
                print(f"Warning: Mask for global frame {global_frame_idx} is empty, skipping.")
    
    print(f"All masks saved to {output_file_path}.")

if __name__ == "__main__":
     main(sys.argv[1:])  # exclude the script name from the args when called from sh
