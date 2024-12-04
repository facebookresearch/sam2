# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import os
from pathlib import Path

import cv2

import numpy as np
import submitit
import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="[SA-V Preprocessing] Extracting JPEG frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------
    # DATA
    # ------------
    data_parser = parser.add_argument_group(
        title="SA-V dataset data root",
        description="What data to load and how to process it.",
    )
    data_parser.add_argument(
        "--sav-vid-dir",
        type=str,
        required=True,
        help=("Where to find the SAV videos"),
    )
    data_parser.add_argument(
        "--sav-frame-sample-rate",
        type=int,
        default=4,
        help="Rate at which to sub-sample frames",
    )

    # ------------
    # LAUNCH
    # ------------
    launch_parser = parser.add_argument_group(
        title="Cluster launch settings",
        description="Number of jobs and retry settings.",
    )
    launch_parser.add_argument(
        "--n-jobs",
        type=int,
        required=True,
        help="Shard the run over this many jobs.",
    )
    launch_parser.add_argument(
        "--timeout", type=int, required=True, help="SLURM timeout parameter in minutes."
    )
    launch_parser.add_argument(
        "--partition", type=str, required=True, help="Partition to launch on."
    )
    launch_parser.add_argument(
        "--account", type=str, required=True, help="Partition to launch on."
    )
    launch_parser.add_argument("--qos", type=str, required=True, help="QOS.")

    # ------------
    # OUTPUT
    # ------------
    output_parser = parser.add_argument_group(
        title="Setting for results output", description="Where and how to save results."
    )
    output_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=("Where to dump the extracted jpeg frames"),
    )
    output_parser.add_argument(
        "--slurm-output-root-dir",
        type=str,
        required=True,
        help=("Where to save slurm outputs"),
    )
    return parser


def decode_video(video_path: str):
    assert os.path.exists(video_path)
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)
        else:
            break
    return video_frames


def extract_frames(video_path, sample_rate):
    frames = decode_video(video_path)
    return frames[::sample_rate]


def submitit_launch(video_paths, sample_rate, save_root):
    for path in tqdm.tqdm(video_paths):
        frames = extract_frames(path, sample_rate)
        output_folder = os.path.join(save_root, Path(path).stem)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for fid, frame in enumerate(frames):
            frame_path = os.path.join(output_folder, f"{fid*sample_rate:05d}.jpg")
            cv2.imwrite(frame_path, frame)
    print(f"Saved output to {save_root}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    sav_vid_dir = args.sav_vid_dir
    save_root = args.output_dir
    sample_rate = args.sav_frame_sample_rate

    # List all SA-V videos
    mp4_files = sorted([str(p) for p in Path(sav_vid_dir).glob("*/*.mp4")])
    mp4_files = np.array(mp4_files)
    chunked_mp4_files = [x.tolist() for x in np.array_split(mp4_files, args.n_jobs)]

    print(f"Processing videos in: {sav_vid_dir}")
    print(f"Processing {len(mp4_files)} files")
    print(f"Beginning processing in {args.n_jobs} processes")

    # Submitit params
    jobs_dir = os.path.join(args.slurm_output_root_dir, "%j")
    cpus_per_task = 4
    executor = submitit.AutoExecutor(folder=jobs_dir)
    executor.update_parameters(
        timeout_min=args.timeout,
        gpus_per_node=0,
        tasks_per_node=1,
        slurm_array_parallelism=args.n_jobs,
        cpus_per_task=cpus_per_task,
        slurm_partition=args.partition,
        slurm_account=args.account,
        slurm_qos=args.qos,
    )
    executor.update_parameters(slurm_srun_args=["-vv", "--cpu-bind", "none"])

    # Launch
    jobs = []
    with executor.batch():
        for _, mp4_chunk in tqdm.tqdm(enumerate(chunked_mp4_files)):
            job = executor.submit(
                submitit_launch,
                video_paths=mp4_chunk,
                sample_rate=sample_rate,
                save_root=save_root,
            )
            jobs.append(job)

    for j in jobs:
        print(f"Slurm JobID: {j.job_id}")
    print(f"Saving outputs to {save_root}")
    print(f"Slurm outputs at {args.slurm_output_root_dir}")
