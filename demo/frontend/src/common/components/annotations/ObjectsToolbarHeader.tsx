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
import ToolbarHeaderWrapper from '@/common/components/toolbar/ToolbarHeaderWrapper';
import {isStreamingAtom, streamingStateAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

export default function ObjectsToolbarHeader() {
  const isStreaming = useAtomValue(isStreamingAtom);
  const streamingState = useAtomValue(streamingStateAtom);

  return (
    <ToolbarHeaderWrapper
      title={
        streamingState === 'full'
          ? 'Review tracked objects'
          : isStreaming
            ? 'Tracking objects'
            : 'Select objects'
      }
      description={
        streamingState === 'full'
          ? 'Review your selected objects across the video, and continue to edit if needed. Once everything looks good, press “Next” to continue.'
          : isStreaming
            ? 'Watch the video closely for any places where your objects aren’t tracked correctly. You can also stop tracking to make additional edits.'
            : 'Adjust the selection of your object, or add additional objects. Press “Track objects” to track your objects throughout the video.'
      }
      className="mb-8"
    />
  );
}
