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
import {useAtom} from 'jotai';
import {useEffect, useRef} from 'react';
import {Message, messageAtom} from '@/common/components/snackbar/snackbarAtoms';

export default function useExpireMessage() {
  const [message, setMessage] = useAtom(messageAtom);
  const messageRef = useRef<Message | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    messageRef.current = message;
  }, [message]);

  useEffect(() => {
    function resetInterval() {
      if (intervalRef.current != null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    if (intervalRef.current == null && message != null && message.expire) {
      intervalRef.current = setInterval(() => {
        const prevMessage = messageRef.current;
        if (prevMessage == null) {
          setMessage(null);
          resetInterval();
          return;
        }
        const messageDuration = Date.now() - prevMessage.startTime;
        if (messageDuration > prevMessage.duration) {
          setMessage(null);
          resetInterval();
          return;
        }
        setMessage({
          ...prevMessage,
          progress: messageDuration / prevMessage.duration,
        });
      }, 20);
    }
  }, [message, setMessage]);

  useEffect(() => {
    return () => {
      if (intervalRef.current != null) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
}
