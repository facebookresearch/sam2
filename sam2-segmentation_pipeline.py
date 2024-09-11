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

    print(isinstance(df.columns, pd.MultiIndex))
    print(list(df.columns))

    return df

def preprocess_image(img):
    # Check if the image is grayscale (1 channel)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Convert grayscale to RGB
        img_rgb = Image.fromarray(img).convert('RGB')
        img_np = np.array(img_rgb)
    else:
        img_np = img

    # Ensure the image is in RGB format
    if img_np.shape[2] != 3:
        raise ValueError("Image must be in RGB format")

    # The SAM model expects the image in its original format (no normalization needed)
    return img_np

def segment_object(pil_image, points, predictor):
    print("Pointer Coordinates:", points)
    # Convert PIL Image to numpy array
    image_np = np.array(pil_image)

    # Preprocess the image
    image_preprocessed = preprocess_image(image_np)

    # Set the image for the predictor
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(image_preprocessed)

        # Prepare the input prompts (points and labels)
        input_points = np.array(points)  # points should be a list of (x, y) tuples
        input_labels = np.ones(len(points))  # 1 for each point, indicating foreground

        # Predict the mask using multiple points
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

    return masks[0]  # Return the first (and only) mask

def extract_coordinates(data, bodyparts):
    result = {}
    for bodypart in bodyparts:
        if bodypart in data.columns.get_level_values(0):
            x_values = pd.to_numeric(data[bodypart]['x'], errors='coerce')
            y_values = pd.to_numeric(data[bodypart]['y'], errors='coerce')
            result[bodypart] = list(zip(x_values, y_values))
    return result

def main(args):

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process input files and generate output.")
    parser.add_argument("input_file_path", type=str, help="Path to the input file")
    parser.add_argument("output_file_path", type=str, help="Path to the output file")
    parser.add_argument("DLC_csv_file_path", type=str, help="Path to the DLC CSV file")
    parser.add_argument("column_names", type=str, nargs='+', help="List of column names")
    parser.add_argument("SAM2_path", type=str, help="Location of GitRepo!")

    # Parse the arguments
    args = parser.parse_args(args)

    SAM2_path = "/lisc/scratch/neurobiology/zimmer/schaar/code/github/segment-anything-2"

    # Define paths relative to the base path
    checkpoint = os.path.join(SAM2_path , "checkpoints", "sam2_hiera_large.pt")
    model_cfg = "/" + os.path.join(SAM2_path, "sam2_configs", "sam2_hiera_l.yaml")

    # Check if files exist
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Config file not found: {model_cfg}")

    # If files exist, proceed with loading the model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # Access the parsed arguments
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    DLC_csv_file_path = args.DLC_csv_file_path
    column_names = args.column_names

    # TODO: Add your main logic here
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"DLC CSV file: {DLC_csv_file_path}")
    print(f"Column names: {column_names}")

    DLC_data = read_DLC_csv(DLC_csv_file_path)

    print(DLC_data.head())

    extracted_coordinates = extract_coordinates(DLC_data, column_names)

    [print(f"{col}:\n{extracted_coordinates.get(col, 'No data available')}") for col in column_names]

    try:
        if os.path.isdir(input_file_path):
            reader_obj = MicroscopeDataReader(input_file_path)
        elif os.path.isfile(input_file_path):
            reader_obj = MicroscopeDataReader(input_file_path, as_raw_tiff=True, raw_tiff_num_slices=1)
        else:
            raise ValueError("Invalid input file path. Please provide a valid directory or file path.")

        tif = da.squeeze(reader_obj.dask_array)

        with tiff.TiffWriter(output_file_path, bigtiff=True) as tif_writer:
            total_frames = len(tif)
            for i, img in enumerate(tif):
                logging.info(f"Processing image {i+1}/{total_frames}")
                img = np.array(img)

                mask = segment_object(img, [extracted_coordinates[column][i] for column in column_names], predictor)

                tif_writer.write(mask, contiguous=True)
                
                logging.info(f"Successfully processed and wrote image {i+1}/{total_frames}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main(sys.argv[1:])  # exclude the script name from the args when called from sh
