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
import {EnqueueOption} from '@/common/components/snackbar/useMessagesSnackbar';

export type MessageOptions = EnqueueOption & {
  repeat?: boolean;
};

type MessageEvent = {
  text: string;
  shown: boolean;
  action?: Element;
  options?: MessageOptions;
};

export interface MessagesEventMap {
  startSession: MessageEvent;
  firstClick: MessageEvent;
  pointClick: MessageEvent;
  addObjectClick: MessageEvent;
  trackAndPlayClick: MessageEvent;
  trackAndPlayComplete: MessageEvent;
  trackAndPlayThrottlingWarning: MessageEvent;
  effectsMessage: MessageEvent;
}

export const defaultMessageMap: MessagesEventMap = {
  startSession: {
    text: 'Starting session',
    shown: false,
    options: {type: 'loading', showClose: false, repeat: true, duration: 2000},
  },
  firstClick: {
    text: 'Tip: Click on any object in the video to get started.',
    shown: false,
    options: {expire: false, repeat: false},
  },
  pointClick: {
    text: 'Tip: Not what you expected? Add a few more clicks until the full object you want is selected.',
    shown: false,
    options: {expire: false, repeat: false},
  },
  addObjectClick: {
    text: 'Tip: Add a new object by clicking on it in the video.',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayClick: {
    text: 'Hang tight while your objects are tracked! You’ll be able to apply visual effects in the next step. Stop tracking at any point to adjust your selections if the tracking doesn’t look right.',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayComplete: {
    text: 'Tip: You can fix tracking issues by going back to the frames where tracking is not quite right and adding or removing clicks.',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayThrottlingWarning: {
    text: 'Looks like you have clicked the tracking button a bit too often! To keep things running smoothly, we have temporarily disabled the button.',
    shown: false,
    options: {repeat: true},
  },
  effectsMessage: {
    text: 'Tip: If you aren’t sure where to get started, click “Surprise Me” to apply a surprise effect to your video.',
    shown: false,
    options: {expire: false, repeat: false},
  },
};
