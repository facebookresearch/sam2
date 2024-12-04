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
import {activeTrackletObjectIdAtom, labelTypeAtom} from '@/demo/atoms';
import {Add} from '@carbon/icons-react';
import {useSetAtom} from 'jotai';

export default function AddObjectButton() {
  const video = useVideo();
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const setLabelType = useSetAtom(labelTypeAtom);
  const {enqueueMessage} = useMessagesSnackbar();

  async function addObject() {
    enqueueMessage('addObjectClick');
    const tracklet = await video?.createTracklet();
    if (tracklet != null) {
      setActiveTrackletId(tracklet.id);
      setLabelType('positive');
    }
  }

  return (
    <div
      onClick={addObject}
      className="group flex justify-start mx-4 px-4 bg-transparent text-white !rounded-xl border-none cursor-pointer">
      <div className="flex gap-6 items-center">
        <div className=" group-hover:bg-graydark-700 border border-white relative h-12 w-12 md:w-20 md:h-20 shrink-0 rounded-lg flex items-center justify-center">
          <Add size={36} className="group-hover:text-white text-gray-300" />
        </div>
        <div className="font-medium text-base">Add another object</div>
      </div>
    </div>
  );
}
