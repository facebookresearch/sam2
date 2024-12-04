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

export type Settings = {
  videoAPIEndpoint: string;
  inferenceAPIEndpoint: string;
};

// Key used to store the settings in the browser's local storage.
export const SAM2_SETTINGS_KEY = 'SAM2_SETTINGS_KEY';

export type Action =
  | {type: 'load-state'}
  | {type: 'change-video-api-endpoint'; url: string}
  | {type: 'change-inference-api-endpoint'; url: string};

export const DEFAULT_SETTINGS: Settings = {
  videoAPIEndpoint: VIDEO_API_ENDPOINT,
  inferenceAPIEndpoint: INFERENCE_API_ENDPOINT,
};

export function settingsReducer(state: Settings, action: Action): Settings {
  function storeSettings(newState: Settings): void {
    localStorage.setItem(SAM2_SETTINGS_KEY, JSON.stringify(newState));
  }

  switch (action.type) {
    case 'load-state': {
      try {
        const serializedSettings = localStorage.getItem(SAM2_SETTINGS_KEY);
        if (serializedSettings != null) {
          return JSON.parse(serializedSettings) as Settings;
        } else {
          // Store default settings in local storage. This will populate the
          // settings in the local storage on first app load or when user
          // cleared the browser cache.
          storeSettings(state);
        }
      } catch {
        // Could not parse settings. Using default settings instead.
      }
      return state;
    }
    case 'change-video-api-endpoint':
      state.videoAPIEndpoint = action.url;
      break;
    case 'change-inference-api-endpoint':
      state.inferenceAPIEndpoint = action.url;
      break;
  }

  // Store the settings state on every change
  storeSettings(state);

  return state;
}
