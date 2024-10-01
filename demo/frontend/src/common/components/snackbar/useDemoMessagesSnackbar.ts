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
import {MessagesEventMap} from '@/common/components/snackbar/DemoMessagesSnackbarUtils';
import useMessagesSnackbar from '@/common/components/snackbar/useMessagesSnackbar';
import {messageMapAtom} from '@/demo/atoms';
import {useAtom} from 'jotai';
import {useCallback} from 'react';

type State = {
  enqueueMessage: (messageType: keyof MessagesEventMap) => void;
  clearMessage: () => void;
};

export default function useDemoMessagesSnackbar(): State {
  const [messageMap, setMessageMap] = useAtom(messageMapAtom);
  const {enqueueMessage: enqueue, clearMessage} = useMessagesSnackbar();

  const enqueueMessage = useCallback(
    (messageType: keyof MessagesEventMap) => {
      const {text, shown, options} = messageMap[messageType];
      if (!options?.repeat && shown === true) {
        return;
      }
      enqueue(text, options);
      const newState = {...messageMap};
      newState[messageType].shown = true;
      setMessageMap(newState);
    },
    [enqueue, messageMap, setMessageMap],
  );

  return {enqueueMessage, clearMessage};
}
