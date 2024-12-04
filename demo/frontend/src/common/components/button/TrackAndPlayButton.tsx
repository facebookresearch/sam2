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
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useFunctionThrottle from '@/common/components/useFunctionThrottle';
import useVideo from '@/common/components/video/editor/useVideo';
import {
  areTrackletObjectsInitializedAtom,
  isStreamingAtom,
  sessionAtom,
  streamingStateAtom,
} from '@/demo/atoms';
import {ChevronRight} from '@carbon/icons-react';
import {useAtom, useAtomValue, useSetAtom} from 'jotai';
import {useCallback, useEffect} from 'react';

export default function TrackAndPlayButton() {
  const video = useVideo();
  const [isStreaming, setIsStreaming] = useAtom(isStreamingAtom);
  const streamingState = useAtomValue(streamingStateAtom);
  const areObjectsInitialized = useAtomValue(areTrackletObjectsInitializedAtom);
  const setSession = useSetAtom(sessionAtom);
  const {enqueueMessage} = useMessagesSnackbar();
  const {isThrottled, maxThrottles, throttle} = useFunctionThrottle(250, 4);

  const isTrackAndPlayDisabled =
    streamingState === 'aborting' || streamingState === 'requesting';

  useEffect(() => {
    function onStreamingStarted() {
      setIsStreaming(true);
    }
    video?.addEventListener('streamingStarted', onStreamingStarted);

    function onStreamingCompleted() {
      enqueueMessage('trackAndPlayComplete');
      setIsStreaming(false);
    }
    video?.addEventListener('streamingCompleted', onStreamingCompleted);

    return () => {
      video?.removeEventListener('streamingStarted', onStreamingStarted);
      video?.removeEventListener('streamingCompleted', onStreamingCompleted);
    };
  }, [video, setIsStreaming, enqueueMessage]);

  const handleTrackAndPlay = useCallback(() => {
    if (isTrackAndPlayDisabled) {
      return;
    }
    if (maxThrottles && isThrottled) {
      enqueueMessage('trackAndPlayThrottlingWarning');
    }

    // Throttling is only applied while streaming because we should
    // only throttle after a user has aborted inference. This way,
    // a user can still quickly abort a stream if they notice the
    // inferred mask is misaligned.
    throttle(
      () => {
        if (!isStreaming) {
          enqueueMessage('trackAndPlayClick');
          video?.streamMasks();
          setSession(previousSession =>
            previousSession == null
              ? previousSession
              : {...previousSession, ranPropagation: true},
          );
        } else {
          video?.abortStreamMasks();
        }
      },
      {enableThrottling: isStreaming},
    );
  }, [
    isTrackAndPlayDisabled,
    isThrottled,
    isStreaming,
    maxThrottles,
    video,
    setSession,
    enqueueMessage,
    throttle,
  ]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const callback = {
        KeyK: handleTrackAndPlay,
      }[event.code];
      if (callback != null) {
        event.preventDefault();
        callback();
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('keydown', handleKey);
    };
  }, [handleTrackAndPlay]);

  return (
    <PrimaryCTAButton
      disabled={isThrottled || !areObjectsInitialized}
      onClick={handleTrackAndPlay}
      endIcon={isStreaming ? undefined : <ChevronRight size={20} />}>
      {isStreaming ? 'Cancel Tracking' : 'Track objects'}
    </PrimaryCTAButton>
  );
}
