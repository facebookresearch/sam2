# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable


def resolve_videos(node_ids: Iterable[str], required: bool = False):
    """
    Resolve videos given node ids.
    """
    from data.store import get_videos

    all_videos = get_videos()
    return [
        all_videos[nid] if required else all_videos.get(nid, None) for nid in node_ids
    ]
