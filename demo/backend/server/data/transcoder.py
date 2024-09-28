# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import av
from app_conf import FFMPEG_NUM_THREADS
from dataclasses_json import dataclass_json

TRANSCODE_VERSION = 1


@dataclass_json
@dataclass
class VideoMetadata:
    duration_sec: Optional[float]
    video_duration_sec: Optional[float]
    container_duration_sec: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    num_video_frames: int
    num_video_streams: int
    video_start_time: float


def transcode(
    in_path: str,
    out_path: str,
    in_metadata: Optional[VideoMetadata],
    seek_t: float,
    duration_time_sec: float,
):
    codec = os.environ.get("VIDEO_ENCODE_CODEC", "libx264")
    crf = int(os.environ.get("VIDEO_ENCODE_CRF", "23"))
    fps = int(os.environ.get("VIDEO_ENCODE_FPS", "24"))
    max_w = int(os.environ.get("VIDEO_ENCODE_MAX_WIDTH", "1280"))
    max_h = int(os.environ.get("VIDEO_ENCODE_MAX_HEIGHT", "720"))
    verbose = ast.literal_eval(os.environ.get("VIDEO_ENCODE_VERBOSE", "False"))

    normalize_video(
        in_path=in_path,
        out_path=out_path,
        max_w=max_w,
        max_h=max_h,
        seek_t=seek_t,
        max_time=duration_time_sec,
        in_metadata=in_metadata,
        codec=codec,
        crf=crf,
        fps=fps,
        verbose=verbose,
    )


def get_video_metadata(path: str) -> VideoMetadata:
    with av.open(path) as cont:
        num_video_streams = len(cont.streams.video)
        width, height, fps = None, None, None
        video_duration_sec = 0
        container_duration_sec = float((cont.duration or 0) / av.time_base)
        video_start_time = 0.0
        rotation_deg = 0
        num_video_frames = 0
        if num_video_streams > 0:
            video_stream = cont.streams.video[0]
            assert video_stream.time_base is not None

            # for rotation, see: https://github.com/PyAV-Org/PyAV/pull/1249
            rotation_deg = video_stream.side_data.get("DISPLAYMATRIX", 0)
            num_video_frames = video_stream.frames
            video_start_time = float(video_stream.start_time * video_stream.time_base)
            width, height = video_stream.width, video_stream.height
            fps = float(video_stream.guessed_rate)
            fps_avg = video_stream.average_rate
            if video_stream.duration is not None:
                video_duration_sec = float(
                    video_stream.duration * video_stream.time_base
                )
            if fps is None:
                fps = float(fps_avg)

            if not math.isnan(rotation_deg) and int(rotation_deg) in (
                90,
                -90,
                270,
                -270,
            ):
                width, height = height, width

        duration_sec = max(container_duration_sec, video_duration_sec)

        return VideoMetadata(
            duration_sec=duration_sec,
            container_duration_sec=container_duration_sec,
            video_duration_sec=video_duration_sec,
            video_start_time=video_start_time,
            fps=fps,
            width=width,
            height=height,
            num_video_streams=num_video_streams,
            num_video_frames=num_video_frames,
        )


def normalize_video(
    in_path: str,
    out_path: str,
    max_w: int,
    max_h: int,
    seek_t: float,
    max_time: float,
    in_metadata: Optional[VideoMetadata],
    codec: str = "libx264",
    crf: int = 23,
    fps: int = 24,
    verbose: bool = False,
):
    if in_metadata is None:
        in_metadata = get_video_metadata(in_path)

    assert in_metadata.num_video_streams > 0, "no video stream present"

    w, h = in_metadata.width, in_metadata.height
    assert w is not None, "width not available"
    assert h is not None, "height not available"

    # rescale to max_w:max_h if needed & preserve aspect ratio
    r = w / h
    if r < 1:
        h = min(720, h)
        w = h * r
    else:
        w = min(1280, w)
        h = w / r

    # h264 cannot encode w/ odd dimensions
    w = int(w)
    h = int(h)
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1

    ffmpeg = shutil.which("ffmpeg")
    cmd = [
        ffmpeg,
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # global threads
        "-ss",
        f"{seek_t:.2f}",
        "-t",
        f"{max_time:.2f}",
        "-i",
        in_path,
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # decode (or filter..?) threads
        "-vf",
        f"fps={fps},scale={w}:{h},setsar=1:1",
        "-c:v",
        codec,
        "-crf",
        f"{crf}",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # encode threads
        out_path,
        "-y",
    ]
    if verbose:
        print(" ".join(cmd))

    subprocess.call(
        cmd,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )
