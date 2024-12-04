# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import imagesize
from app_conf import GALLERY_PATH, POSTERS_PATH, POSTERS_PREFIX
from data.data_types import Video
from tqdm import tqdm


def preload_data() -> Dict[str, Video]:
    """
    Preload data including gallery videos and their posters.
    """
    # Dictionaries for videos and datasets on the backend.
    # Note that since Python 3.7, dictionaries preserve their insert order, so
    # when looping over its `.values()`, elements inserted first also appear first.
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    all_videos = {}

    video_path_pattern = os.path.join(GALLERY_PATH, "**/*.mp4")
    video_paths = glob(video_path_pattern, recursive=True)

    for p in tqdm(video_paths):
        video = get_video(p, GALLERY_PATH)
        all_videos[video.code] = video

    return all_videos


def get_video(
    filepath: os.PathLike,
    absolute_path: Path,
    file_key: Optional[str] = None,
    generate_poster: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> Video:
    """
    Get video object given
    """
    # Use absolute_path to include the parent directory in the video
    video_path = os.path.relpath(filepath, absolute_path.parent)
    poster_path = None
    if generate_poster:
        poster_id = os.path.splitext(os.path.basename(filepath))[0]
        poster_filename = f"{str(poster_id)}.jpg"
        poster_path = f"{POSTERS_PREFIX}/{poster_filename}"

        # Extract the first frame from video
        poster_output_path = os.path.join(POSTERS_PATH, poster_filename)
        ffmpeg = shutil.which("ffmpeg")
        subprocess.call(
            [
                ffmpeg,
                "-y",
                "-i",
                str(filepath),
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                "1",
                "-update",
                "1",
                "-strict",
                "unofficial",
                str(poster_output_path),
            ],
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        # Extract video width and height from poster. This is important to optimize
        # rendering previews in the mosaic video preview.
        width, height = imagesize.get(poster_output_path)

    return Video(
        code=video_path,
        path=video_path if file_key is None else file_key,
        poster_path=poster_path,
        width=width,
        height=height,
    )
