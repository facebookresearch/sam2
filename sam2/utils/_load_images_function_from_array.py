import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_video_frames_from_array(
    image_array,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """
    Load video frames from an array of images.

    The frames are resized to `image_size x image_size` and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    Args:
    - image_array: List or NumPy array of images in the shape (num_frames, H, W, 3) (RGB images).
    - image_size: Target image size (the frames will be resized to square resolution: image_size x image_size).
    - offload_video_to_cpu: Boolean flag to offload the frames to CPU.
    - img_mean: Mean for image normalization (default is for ImageNet).
    - img_std: Standard deviation for image normalization (default is for ImageNet).
    - compute_device: Device to load the frames (e.g., "cuda" for GPU, "cpu" for CPU).

    Returns:
    - images: A tensor of shape (num_frames, 3, image_size, image_size) containing the resized and normalized images.
    - video_height: The original height of the images.
    - video_width: The original width of the images.
    """
    if isinstance(image_array, np.ndarray) and len(image_array.shape) == 4:
        # Expecting image_array in shape (num_frames, height, width, 3)
        num_frames, video_height, video_width, _ = image_array.shape
    else:
        raise RuntimeError("Expected image_array to be a 4D NumPy array (num_frames, height, width, 3)")

    # Convert mean and std to tensors
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # Initialize a tensor to hold the resized images
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)

    for n in tqdm(range(num_frames), desc="Processing frames"):
        # Convert each frame from NumPy array to PIL Image
        img_np = image_array[n]  # shape (H, W, 3)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))  # Convert to 0-255 for PIL compatibility

        # Resize the image to the target size
        img_resized = img_pil.resize((image_size, image_size))

        # Convert the resized image to a torch tensor and normalize to range [0, 1]
        img_tensor = torch.from_numpy(np.array(img_resized).astype(np.float32) / 255.0).permute(2, 0, 1)  # HWC -> CHW

        # Place the processed tensor into the larger tensor
        images[n] = img_tensor

    # Move images to the desired device (CPU or GPU)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    # Normalize images by subtracting the mean and dividing by the standard deviation
    images = (images - img_mean) / img_std

    return images, video_height, video_width
