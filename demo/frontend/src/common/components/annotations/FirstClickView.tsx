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
import ChangeVideo from '@/common/components/gallery/ChangeVideoModal';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import {DEMO_SHORT_NAME} from '@/demo/DemoConfig';
import {useEffect, useRef} from 'react';

export default function FirstClickView() {
  const isFirstClickMessageShown = useRef(false);
  const {enqueueMessage} = useMessagesSnackbar();

  useEffect(() => {
    if (!isFirstClickMessageShown.current) {
      isFirstClickMessageShown.current = true;
      enqueueMessage('firstClick');
    }
  }, [enqueueMessage]);

  return (
    <div className="w-full h-full flex flex-col p-8">
      <div className="grow flex flex-col gap-6">
        <h2 className="text-2xl">Click an object in the video to start</h2>
        <p className="!text-gray-60">
          You&apos;ll be able to use {DEMO_SHORT_NAME} to make fun edits to any
          video by tracking objects and applying visual effects.
        </p>
        <p className="!text-gray-60">
          To start, click any object in the video.
        </p>
      </div>
      <div className="flex items-center">
        <ChangeVideo />
      </div>
    </div>
  );
}
