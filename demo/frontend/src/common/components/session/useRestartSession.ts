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
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import useInputVideo from '@/common/components/video/useInputVideo';
import {
  activeTrackletObjectIdAtom,
  isPlayingAtom,
  isStreamingAtom,
  labelTypeAtom,
  trackletObjectsAtom,
} from '@/demo/atoms';
import {useAtomValue, useSetAtom} from 'jotai';
import {useState} from 'react';

export default function useRestartSession() {
  const [isLoading, setIsLoading] = useState<boolean>();
  const isPlaying = useAtomValue(isPlayingAtom);
  const isStreaming = useAtomValue(isStreamingAtom);
  const setActiveTrackletObjectId = useSetAtom(activeTrackletObjectIdAtom);
  const setTracklets = useSetAtom(trackletObjectsAtom);
  const setLabelType = useSetAtom(labelTypeAtom);
  const {clearMessage} = useMessagesSnackbar();

  const {inputVideo} = useInputVideo();
  const video = useVideo();

  async function restartSession(onRestart?: () => void) {
    if (video === null || inputVideo === null) {
      return;
    }

    setIsLoading(true);
    if (isPlaying) {
      video.pause();
    }
    if (isStreaming) {
      await video.abortStreamMasks();
    }
    await video?.startSession(inputVideo.path);
    video.frame = 0;
    setActiveTrackletObjectId(0);
    setTracklets([]);
    setLabelType('positive');
    onRestart?.();
    clearMessage();
    setIsLoading(false);
  }

  return {isLoading, restartSession};
}
