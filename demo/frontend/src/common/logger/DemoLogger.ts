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
import {RenderingErrorType} from '@/common/error/ErrorUtils';
import Logger from './Logger';

type UploadSourceType = 'gallery' | 'option';

// Maps event names to an optional payload for each event
type DemoEventMap = {
  // User events
  user_click_canvas: {
    click_type: 'add_point' | 'remove_point';
    click_action: 'add_object' | 'refine_object';
    click_variant?: 'positive' | 'negative';
  };
  user_click_object: {
    tracklet_id: number;
  };
  user_click_track_and_play: {
    track_and_play_click_type: 'stream' | 'abort';
  };
  user_click_apply_effect: {
    effect_type: 'background' | 'object';
    effect_name: string;
    effect_variant: number;
  };
  user_change_video: {
    gallery_video_url: string;
  };
  user_upload_video: {
    upload_source: UploadSourceType;
  };
  user_click_share: {
    gallery_video_url: string;
  };
  user_click_download: {
    gallery_video_url: string;
  };
  user_click_web_share: undefined;
  // Error events
  client_error_rendering: {
    rendering_error_type: RenderingErrorType;
  };
  client_error_start_session: undefined;
  client_error_upload_video: {
    upload_source: UploadSourceType;
    upload_error_message: string;
  };
  client_error_unsupported_browser: undefined;
  client_error_page_not_found: {
    path: string;
  };
  client_error_general: {
    message: string;
  };
  client_error_fallback: {
    fallback_error_message: string;
  };

  // Dataset events
  client_error_fallback_dataset: {
    dataset_fallback_error_message: string;
  };
  dataset_client_impression_event: {
    impression_type: 'grid_view' | 'detailed_view';
    video_id?: string;
  };
  dataset_client_click_events: {
    click_type: 'search' | 'next_page' | 'prev_page';
    video_id?: string;
  };
};

export interface LoggerInterface<TEventMap> {
  event: <K extends keyof TEventMap>(
    eventName: K,
    options?: TEventMap[K],
  ) => void;
}

export function initialize(): void {
  // noop
}

export class DemoLogger implements LoggerInterface<DemoEventMap> {
  event<K extends keyof DemoEventMap>(eventName: K, options?: DemoEventMap[K]) {
    Logger.info(eventName, options ?? {});
  }
}
