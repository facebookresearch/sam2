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
import {SegmentationPoint} from '@/common/tracker/Tracker';
import {TrackerOptions, Trackers} from '@/common/tracker/Trackers';
import {
  AddPointsEvent,
  ClearPointsInVideoEvent,
  SessionStartFailedEvent,
  SessionStartedEvent,
  StreamingCompletedEvent,
  StreamingStartedEvent,
  StreamingStateUpdateEvent,
  TrackletCreatedEvent,
  TrackletDeletedEvent,
  TrackletsEvent,
} from '../components/video/VideoWorkerBridge';

export type Flags = {
  masks: boolean;
  effect: boolean;
};

export type Request<A, P> = {
  action: A;
} & P;

// REQUESTS

export type InitializeTrackerRequest = Request<
  'initializeTracker',
  {
    name: keyof Trackers;
    options: TrackerOptions;
  }
>;
export type StartSessionRequest = Request<
  'startSession',
  {
    videoUrl: string;
  }
>;
export type CloseSessionRequest = Request<'closeSession', unknown>;
export type CreateTrackletRequest = Request<'createTracklet', unknown>;
export type DeleteTrackletRequest = Request<
  'deleteTracklet',
  {
    trackletId: number;
  }
>;
export type UpdatePointsRequest = Request<
  'updatePoints',
  {
    frameIndex: number;
    objectId: number;
    points: SegmentationPoint[];
  }
>;
export type ClearPointsInFrameRequest = Request<
  'clearPointsInFrame',
  {
    frameIndex: number;
    objectId: number;
  }
>;
export type ClearPointsInVideoRequest = Request<'clearPointsInVideo', unknown>;
export type StreamMasksRequest = Request<
  'streamMasks',
  {
    frameIndex: number;
  }
>;
export type AbortStreamMasksRequest = Request<'abortStreamMasks', unknown>;

export type LogAnnotationsRequest = Request<'logAnnotations', unknown>;

export type TrackerRequest =
  | InitializeTrackerRequest
  | StartSessionRequest
  | CloseSessionRequest
  | CreateTrackletRequest
  | DeleteTrackletRequest
  | UpdatePointsRequest
  | ClearPointsInFrameRequest
  | ClearPointsInVideoRequest
  | StreamMasksRequest
  | AbortStreamMasksRequest
  | LogAnnotationsRequest;

export type TrackerRequestMessageEvent = MessageEvent<TrackerRequest>;

// RESPONSES

export type SessionStartedResponse = Request<
  'sessionStarted',
  SessionStartedEvent
>;

export type SessionStartFailedResponse = Request<
  'sessionStartFailed',
  SessionStartFailedEvent
>;

export type TrackletCreatedResponse = Request<
  'trackletCreated',
  TrackletCreatedEvent
>;

export type TrackletsUpdatedResponse = Request<
  'trackletsUpdated',
  TrackletsEvent
>;

export type TrackletDeletedResponse = Request<
  'trackletDeleted',
  TrackletDeletedEvent
>;

export type AddPointsResponse = Request<'addPoints', AddPointsEvent>;

export type ClearPointsInVideoResponse = Request<
  'clearPointsInVideo',
  ClearPointsInVideoEvent
>;

export type StreamingStartedResponse = Request<
  'streamingStarted',
  StreamingStartedEvent
>;

export type StreamingCompletedResponse = Request<
  'streamingCompleted',
  StreamingCompletedEvent
>;

export type StreamingStateUpdateResponse = Request<
  'streamingStateUpdate',
  StreamingStateUpdateEvent
>;

export type TrackerResponse =
  | SessionStartedResponse
  | SessionStartFailedResponse
  | TrackletCreatedResponse
  | TrackletsUpdatedResponse
  | TrackletDeletedResponse
  | AddPointsResponse
  | ClearPointsInVideoResponse
  | StreamingStartedResponse
  | StreamingCompletedResponse
  | StreamingStateUpdateResponse;

export type TrackerResponseMessageEvent = MessageEvent<TrackerResponse>;
