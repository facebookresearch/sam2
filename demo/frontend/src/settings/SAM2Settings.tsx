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
import {INFERENCE_API_ENDPOINT, VIDEO_API_ENDPOINT} from '@/demo/DemoConfig';
import ApprovableInput from '@/settings/ApprovableInput';
import useSettingsContext from '@/settings/useSettingsContext';

export default function SAMVSettings() {
  const {settings, dispatch} = useSettingsContext();

  return (
    <div>
      <ApprovableInput
        label="Video API Endpoint"
        defaultValue={VIDEO_API_ENDPOINT}
        initialValue={settings.videoAPIEndpoint}
        onChange={url => dispatch({type: 'change-video-api-endpoint', url})}
      />
      <ApprovableInput
        label="Inference API Endpoint"
        defaultValue={INFERENCE_API_ENDPOINT}
        initialValue={settings.inferenceAPIEndpoint}
        onChange={url => dispatch({type: 'change-inference-api-endpoint', url})}
      />
    </div>
  );
}
