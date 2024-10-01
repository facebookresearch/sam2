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
import {useSetAtom} from 'jotai';
import {useCallback} from 'react';
import {
  MessageType,
  messageAtom,
} from '@/common/components/snackbar/snackbarAtoms';

export type EnqueueOption = {
  duration?: number;
  type?: MessageType;
  expire?: boolean;
  showClose?: boolean;
  showReset?: boolean;
};

type State = {
  clearMessage: () => void;
  enqueueMessage: (message: string, options?: EnqueueOption) => void;
};

export default function useMessagesSnackbar(): State {
  const setMessage = useSetAtom(messageAtom);

  const enqueueMessage = useCallback(
    (message: string, options?: EnqueueOption) => {
      setMessage({
        text: message,
        type: options?.type ?? 'info',
        duration: options?.duration ?? 5000,
        progress: 0,
        startTime: Date.now(),
        expire: options?.expire ?? true,
        showClose: options?.showClose ?? true,
        showReset: options?.showReset ?? false,
      });
    },
    [setMessage],
  );

  function clearMessage() {
    setMessage(null);
  }

  return {enqueueMessage, clearMessage};
}
