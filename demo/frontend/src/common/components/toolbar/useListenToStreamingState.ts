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
import {StreamingStateUpdateEvent} from '@/common/components/video/VideoWorkerBridge';
import useVideo from '@/common/components/video/editor/useVideo';
import {StreamingState} from '@/common/tracker/Tracker';
import {isStreamingAtom, streamingStateAtom} from '@/demo/atoms';
import {useAtom} from 'jotai';
import {useEffect} from 'react';

export default function useListenToStreamingState(): {
  isStreaming: boolean;
  streamingState: StreamingState;
} {
  const [streamingState, setStreamingState] = useAtom(streamingStateAtom);
  const [isStreaming, setIsStreaming] = useAtom(isStreamingAtom);
  const video = useVideo();

  useEffect(() => {
    function onStreamingStateUpdate(event: StreamingStateUpdateEvent) {
      setStreamingState(event.state);
    }
    function onStreamingStarted() {
      setIsStreaming(true);
    }
    function onStreamingCompleted() {
      setIsStreaming(false);
    }
    video?.addEventListener('streamingStateUpdate', onStreamingStateUpdate);
    video?.addEventListener('streamingStarted', onStreamingStarted);
    video?.addEventListener('streamingCompleted', onStreamingCompleted);

    return () => {
      video?.removeEventListener(
        'streamingStateUpdate',
        onStreamingStateUpdate,
      );
      video?.removeEventListener('streamingStarted', onStreamingStarted);
      video?.removeEventListener('streamingCompleted', onStreamingCompleted);
    };
  }, [video, setStreamingState, setIsStreaming]);

  return {isStreaming, streamingState};
}
