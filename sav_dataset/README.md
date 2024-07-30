# Segment Anything Video (SA-V) Dataset

## Overview

[Segment Anything Video (SA-V)](https://ai.meta.com/datasets/segment-anything-video/), consists of 51K diverse videos and 643K high-quality spatio-temporal segmentation masks (i.e., masklets). The dataset is released under the CC by 4.0 license. Browse the dataset [here](https://sam2.metademolab.com/dataset).

![SA-V dataset](../assets/sa_v_dataset.jpg?raw=true)

## Getting Started

### Download the dataset

Visit [here](https://ai.meta.com/datasets/segment-anything-video-downloads/) to download SA-V including the training, val and test sets.

### Dataset Stats

|            | Num Videos | Num Masklets                              |
| ---------- | ---------- | ----------------------------------------- |
| SA-V train | 50,583     | 642,036 (auto 451,720 and manual 190,316) |
| SA-V val   | 155        | 293                                       |
| SA-V test  | 150        | 278                                       |

### Notebooks

To load and visualize the SA-V training set annotations, refer to the example [sav_visualization_example.ipynb](./sav_visualization_example.ipynb) notebook.

### SA-V train

For SA-V training set we release the mp4 videos and store the masklet annotations per video as json files . Automatic masklets and manual masklets are stored separately as two json files: `{video_id}_auto.json` and `{video_id}_manual.json`. They can be loaded as dictionaries in python in the format below.

```
{
    "video_id"                        : str; video id
    "video_duration"                  : float64; the duration in seconds of this video
    "video_frame_count"               : float64; the number of frames in the video
    "video_height"                    : float64; the height of the video
    "video_width"                     : float64; the width of the video
    "video_resolution"                : float64; video_height $\times$ video_width
    "video_environment"               : List[str]; "Indoor" or "Outdoor"
    "video_split"                     : str; "train" for training set
    "masklet"                         : List[List[Dict]]; masklet annotations in list of list of RLEs.
                                        The outer list is over frames in the video and the inner list
                                        is over objects in the video.
    "masklet_id"                      : List[int]; the masklet ids
    "masklet_size_rel"                : List[float]; the average mask area normalized by resolution
                                        across all the frames where the object is visible
    "masklet_size_abs"                : List[float]; the average mask area (in pixels)
                                        across all the frames where the object is visible
    "masklet_size_bucket"             : List[str]; "small": $1$ <= masklet_size_abs < $32^2$,
                                        "medium": $32^2$ <= masklet_size_abs < $96^2$,
                                        and "large": masklet_size_abs > $96^2$
    "masklet_visibility_changes"      : List[int]; the number of times where the visibility changes
                                        after the first appearance (e.g., invisible -> visible
                                        or visible -> invisible)
    "masklet_first_appeared_frame"    : List[int]; the index of the frame where the object appears
                                        the first time in the video. Always 0 for auto masklets.
    "masklet_frame_count"             : List[int]; the number of frames being annotated. Note that
                                        videos are annotated at 6 fps (annotated every 4 frames)
                                        while the videos are at 24 fps.
    "masklet_edited_frame_count"      : List[int]; the number of frames being edited by human annotators.
                                        Always 0 for auto masklets.
    "masklet_type"                    : List[str]; "auto" or "manual"
    "masklet_stability_score"         : Optional[List[List[float]]]; per-mask stability scores. Auto annotation only.
    "masklet_num"                     : int; the number of manual/auto masklets in the video

}
```

Note that in SA-V train, there are in total 50,583 videos where all of them have manual annotations. Among the 50,583 videos there are 48,436 videos that also have automatic annotations.

### SA-V val and test

For SA-V val and test sets, we release the extracted frames as jpeg files, and the masks as png files with the following directory structure:

```
sav_val(sav_test)
├── sav_val.txt (sav_test.txt): a list of video ids in the split
├── JPEGImages_24fps # videos are extracted at 24 fps
│   ├── {video_id}
│   │     ├── 00000.jpg        # video frame
│   │     ├── 00001.jpg        # video frame
│   │     ├── 00002.jpg        # video frame
│   │     ├── 00003.jpg        # video frame
│   │     └── ...
│   ├── {video_id}
│   ├── {video_id}
│   └── ...
└── Annotations_6fps # videos are annotated at 6 fps
    ├── {video_id}
    │     ├── 000               # obj 000
    │     │    ├── 00000.png    # mask for object 000 in 00000.jpg
    │     │    ├── 00004.png    # mask for object 000 in 00004.jpg
    │     │    ├── 00008.png    # mask for object 000 in 00008.jpg
    │     │    ├── 00012.png    # mask for object 000 in 00012.jpg
    │     │    └── ...
    │     ├── 001               # obj 001
    │     ├── 002               # obj 002
    │     └── ...
    ├── {video_id}
    ├── {video_id}
    └── ...
```

All masklets in val and test sets are manually annotated in every frame by annotators. For each annotated object in a video, we store the annotated masks in a single png. This is because the annotated objects may overlap, e.g., it is possible in our SA-V dataset for there to be a mask for the whole person as well as a separate mask for their hands.

## SA-V Val and Test Evaluation

We provide an evaluator to compute the common J and F metrics on SA-V val and test sets. To run the evaluation, we need to first install a few dependencies as follows:

```
pip install -r requirements.txt
```

Then we can evaluate the predictions as follows:

```
python sav_evaluator.py --gt_root {GT_ROOT} --pred_root {PRED_ROOT}
```

or run

```
python sav_evaluator.py --help
```

to print a complete help message.

The evaluator expects the `GT_ROOT` to be one of the following folder structures, and `GT_ROOT` and `PRED_ROOT` to have the same structure.

- Same as SA-V val and test directory structure

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 000               # all masks associated with obj 000
│     │    ├── 00000.png    # mask for object 000 in frame 00000 (binary mask)
│     │    └── ...
│     ├── 001               # all masks associated with obj 001
│     ├── 002               # all masks associated with obj 002
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

In the paper for the experiments on SA-V val and test, we run inference on the 24 fps videos, and evaluate on the subset of frames where we have ground truth annotations (first and last annotated frames dropped). The evaluator will ignore the masks in frames where we don't have ground truth annotations.

- Same as [DAVIS](https://github.com/davisvideochallenge/davis2017-evaluation) directory structure

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 00000.png        # annotations in frame 00000 (may contain multiple objects)
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

## License

The evaluation code is licensed under the [BSD 3 license](./LICENSE). Please refer to the paper for more details on the models. The videos and annotations in SA-V Dataset are released under CC BY 4.0.

Third-party code: the evaluation software is heavily adapted from [`VOS-Benchmark`](https://github.com/hkchengrex/vos-benchmark) and [`DAVIS`](https://github.com/davisvideochallenge/davis2017-evaluation) (with their licenses in [`LICENSE_DAVIS`](./LICENSE_DAVIS) and [`LICENSE_VOS_BENCHMARK`](./LICENSE_VOS_BENCHMARK)).
