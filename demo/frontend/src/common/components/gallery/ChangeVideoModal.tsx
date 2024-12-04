/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import type {VideoGalleryTriggerProps} from '@/common/components/gallery/DemoVideoGalleryModal';
import DemoVideoGalleryModal from '@/common/components/gallery/DemoVideoGalleryModal';
import useVideo from '@/common/components/video/editor/useVideo';
import Logger from '@/common/logger/Logger';
import {isStreamingAtom, uploadingStateAtom, VideoData} from '@/demo/atoms';
import {useAtomValue, useSetAtom} from 'jotai';
import {ComponentType, useCallback} from 'react';
import {useNavigate} from 'react-router-dom';

type Props = {
  videoGalleryModalTrigger?: ComponentType<VideoGalleryTriggerProps>;
  showUploadInGallery?: boolean;
  onChangeVideo?: () => void;
};

export default function ChangeVideoModal({
  videoGalleryModalTrigger: VideoGalleryModalTriggerComponent,
  showUploadInGallery = true,
  onChangeVideo,
}: Props) {
  const isStreaming = useAtomValue(isStreamingAtom);
  const setUploadingState = useSetAtom(uploadingStateAtom);
  const video = useVideo();
  const navigate = useNavigate();

  const handlePause = useCallback(() => {
    video?.pause();
  }, [video]);

  function handlePauseOrAbortVideo() {
    if (isStreaming) {
      video?.abortStreamMasks();
    } else {
      handlePause();
    }
  }

  function handleSwitchVideos(video: VideoData) {
    // Retain any search parameter
    navigate(
      {
        pathname: location.pathname,
        search: location.search,
      },
      {
        state: {
          video,
        },
      },
    );
    onChangeVideo?.();
  }

  function handleUploadVideoError(error: Error) {
    setUploadingState('error');
    Logger.error(error);
  }

  return (
    <DemoVideoGalleryModal
      trigger={VideoGalleryModalTriggerComponent}
      showUploadInGallery={showUploadInGallery}
      onOpen={handlePauseOrAbortVideo}
      onSelect={handleSwitchVideos}
      onUploadVideoError={handleUploadVideoError}
    />
  );
}
