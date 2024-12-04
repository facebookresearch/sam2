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
import {registerSerializableConstructors} from '@/common/error/ErrorSerializationUtils';
import {Tracker} from '@/common/tracker/Tracker';
import {TrackerRequestMessageEvent} from '@/common/tracker/TrackerTypes';
import {TRACKER_MAPPING} from '@/common/tracker/Trackers';
import {serializeError} from 'serialize-error';
import VideoWorkerContext from './VideoWorkerContext';
import {
  ErrorResponse,
  VideoWorkerRequestMessageEvent,
} from './VideoWorkerTypes';

registerSerializableConstructors();

const context = new VideoWorkerContext();
let tracker: Tracker | null = null;

let statsEnabled = false;

self.addEventListener(
  'message',
  async (
    event: VideoWorkerRequestMessageEvent | TrackerRequestMessageEvent,
  ) => {
    try {
      switch (event.data.action) {
        // Initialize context
        case 'setCanvas':
          context.setCanvas(event.data.canvas);
          break;
        case 'setSource':
          context.setSource(event.data.source);
          break;

        // Playback
        case 'play':
          context.play();
          break;
        case 'pause':
          context.pause();
          break;
        case 'stop':
          context.stop();
          break;
        case 'frameUpdate':
          context.goToFrame(event.data.index);
          break;

        // Filmstrip
        case 'filmstrip': {
          const {width, height} = event.data;
          await context.createFilmstrip(width, height);
          break;
        }

        // Effects
        case 'setEffect': {
          const {name, index, options} = event.data;
          await context.setEffect(name, index, options);
          break;
        }

        // Encode
        case 'encode': {
          await context.encode();
          break;
        }

        case 'enableStats': {
          statsEnabled = true;
          context.enableStats();
          tracker?.enableStats();
          break;
        }

        // Tracker
        case 'initializeTracker': {
          const {name, options} = event.data;
          const Tracker = TRACKER_MAPPING[name];
          // Update the endpoint for the streaming API
          tracker = new Tracker(context, options);
          if (statsEnabled) {
            tracker.enableStats();
          }
          break;
        }
        case 'startSession': {
          const {videoUrl} = event.data;
          await tracker?.startSession(videoUrl);
          break;
        }
        case 'createTracklet':
          tracker?.createTracklet();
          break;
        case 'deleteTracklet':
          await tracker?.deleteTracklet(event.data.trackletId);
          break;
        case 'closeSession':
          tracker?.closeSession();
          break;
        case 'updatePoints': {
          const {frameIndex, objectId, points} = event.data;
          context.allowEffectAnimation(true, objectId, points);
          await tracker?.updatePoints(frameIndex, objectId, points);
          break;
        }
        case 'clearPointsInFrame': {
          const {frameIndex, objectId} = event.data;
          await tracker?.clearPointsInFrame(frameIndex, objectId);
          break;
        }
        case 'clearPointsInVideo':
          await tracker?.clearPointsInVideo();
          break;
        case 'streamMasks': {
          const {frameIndex} = event.data;
          context.allowEffectAnimation(false);
          await tracker?.streamMasks(frameIndex);
          break;
        }
        case 'abortStreamMasks':
          tracker?.abortStreamMasks();
          break;
      }
    } catch (error) {
      const serializedError = serializeError(error);
      const errorResponse: ErrorResponse = {
        action: 'error',
        error: serializedError,
      };
      self.postMessage(errorResponse);
    }
  },
);
