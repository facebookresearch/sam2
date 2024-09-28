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
import {EffectIndex, Effects} from '@/common/components/video/effects/Effects';
import {registerSerializableConstructors} from '@/common/error/ErrorSerializationUtils';
import {
  BaseTracklet,
  SegmentationPoint,
  StreamingState,
} from '@/common/tracker/Tracker';
import {
  AbortStreamMasksRequest,
  AddPointsResponse,
  ClearPointsInFrameRequest,
  ClearPointsInVideoRequest,
  ClearPointsInVideoResponse,
  CloseSessionRequest,
  CreateTrackletRequest,
  DeleteTrackletRequest,
  InitializeTrackerRequest,
  LogAnnotationsRequest,
  SessionStartFailedResponse,
  SessionStartedResponse,
  StartSessionRequest,
  StreamMasksRequest,
  StreamingStateUpdateResponse,
  TrackerRequest,
  TrackerResponseMessageEvent,
  TrackletCreatedResponse,
  TrackletDeletedResponse,
  UpdatePointsRequest,
} from '@/common/tracker/TrackerTypes';
import {TrackerOptions, Trackers} from '@/common/tracker/Trackers';
import {MP4ArrayBuffer} from 'mp4box';
import {deserializeError, type ErrorObject} from 'serialize-error';
import {EventEmitter} from './EventEmitter';
import {
  EncodeVideoRequest,
  FilmstripRequest,
  FilmstripResponse,
  FrameUpdateRequest,
  PauseRequest,
  PlayRequest,
  SetCanvasRequest,
  SetEffectRequest,
  SetSourceRequest,
  StopRequest,
  VideoWorkerRequest,
  VideoWorkerResponseMessageEvent,
} from './VideoWorkerTypes';
import {EffectOptions} from './effects/Effect';

registerSerializableConstructors();

export type DecodeEvent = {
  totalFrames: number;
  numFrames: number;
  fps: number;
  width: number;
  height: number;
  done: boolean;
};

export type LoadStartEvent = unknown;

export type EffectUpdateEvent = {
  name: keyof Effects;
  index: EffectIndex;
  variant: number;
  numVariants: number;
};

export type EncodingStateUpdateEvent = {
  progress: number;
};

export type EncodingCompletedEvent = {
  file: MP4ArrayBuffer;
};

export interface PlayEvent {}

export interface PauseEvent {}

export interface FilmstripEvent {
  filmstrip: ImageBitmap;
}

export interface FrameUpdateEvent {
  index: number;
}

export interface SessionStartedEvent {
  sessionId: string;
}

export interface SessionStartFailedEvent {}

export interface TrackletCreatedEvent {
  // Do not send masks between workers and main thread because they are huge,
  // and sending them would eventually slow down the main thread.
  tracklet: BaseTracklet;
}

export interface TrackletsEvent {
  // Do not send masks between workers and main thread because they are huge,
  // and sending them would eventually slow down the main thread.
  tracklets: BaseTracklet[];
}

export interface TrackletDeletedEvent {
  isSuccessful: boolean;
}

export interface AddPointsEvent {
  isSuccessful: boolean;
}

export interface ClearPointsInVideoEvent {
  isSuccessful: boolean;
}

export interface StreamingStartedEvent {}

export interface StreamingCompletedEvent {}

export interface StreamingStateUpdateEvent {
  state: StreamingState;
}

export interface RenderingErrorEvent {
  error: ErrorObject;
}

export interface VideoWorkerEventMap {
  error: ErrorEvent;
  decode: DecodeEvent;
  encodingStateUpdate: EncodingStateUpdateEvent;
  encodingCompleted: EncodingCompletedEvent;
  play: PlayEvent;
  pause: PauseEvent;
  filmstrip: FilmstripEvent;
  frameUpdate: FrameUpdateEvent;
  sessionStarted: SessionStartedEvent;
  sessionStartFailed: SessionStartFailedEvent;
  trackletCreated: TrackletCreatedEvent;
  trackletsUpdated: TrackletsEvent;
  trackletDeleted: TrackletDeletedEvent;
  addPoints: AddPointsEvent;
  clearPointsInVideo: ClearPointsInVideoEvent;
  streamingStarted: StreamingStartedEvent;
  streamingCompleted: StreamingCompletedEvent;
  streamingStateUpdate: StreamingStateUpdateEvent;
  // HTMLVideoElement events https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video#events
  loadstart: LoadStartEvent;
  effectUpdate: EffectUpdateEvent;
  renderingError: RenderingErrorEvent;
}

type Metadata = {
  totalFrames: number;
  fps: number;
  width: number;
  height: number;
};

export default class VideoWorkerBridge extends EventEmitter<VideoWorkerEventMap> {
  static create(workerFactory: () => Worker) {
    const worker = workerFactory();
    return new VideoWorkerBridge(worker);
  }

  protected worker: Worker;
  private metadata: Metadata | null = null;
  private frameIndex: number = 0;

  private _sessionId: string | null = null;

  public get sessionId() {
    return this._sessionId;
  }

  public get width() {
    return this.metadata?.width ?? 0;
  }

  public get height() {
    return this.metadata?.height ?? 0;
  }

  public get numberOfFrames() {
    return this.metadata?.totalFrames ?? 0;
  }

  public get fps() {
    return this.metadata?.fps ?? 0;
  }

  public get frame() {
    return this.frameIndex;
  }

  constructor(worker: Worker) {
    super();
    this.worker = worker;

    worker.addEventListener(
      'message',
      (
        event: VideoWorkerResponseMessageEvent | TrackerResponseMessageEvent,
      ) => {
        switch (event.data.action) {
          case 'error':
            // Deserialize error before triggering the event
            event.data.error = deserializeError(event.data.error);
            break;
          case 'decode':
            this.metadata = event.data;
            break;
          case 'frameUpdate':
            this.frameIndex = event.data.index;
            break;
          case 'sessionStarted':
            this._sessionId = event.data.sessionId;
            break;
        }
        this.trigger(event.data.action, event.data);
      },
    );
  }

  public setCanvas(canvas: HTMLCanvasElement): void {
    const offscreenCanvas = canvas.transferControlToOffscreen();
    this.sendRequest<SetCanvasRequest>(
      'setCanvas',
      {
        canvas: offscreenCanvas,
      },
      [offscreenCanvas],
    );
  }

  public setSource(source: string): void {
    this.sendRequest<SetSourceRequest>('setSource', {
      source,
    });
  }

  public terminate(): void {
    super.destroy();
    this.worker.terminate();
  }

  public play(): void {
    this.sendRequest<PlayRequest>('play');
  }

  public pause(): void {
    this.sendRequest<PauseRequest>('pause');
  }

  public stop(): void {
    this.sendRequest<StopRequest>('stop');
  }

  public goToFrame(index: number): void {
    this.sendRequest<FrameUpdateRequest>('frameUpdate', {
      index,
    });
  }

  public previousFrame(): void {
    const index = Math.max(0, this.frameIndex - 1);
    this.goToFrame(index);
  }

  public nextFrame(): void {
    const index = Math.min(this.frameIndex + 1, this.numberOfFrames - 1);
    this.goToFrame(index);
  }

  public set frame(index: number) {
    this.sendRequest<FrameUpdateRequest>('frameUpdate', {index});
  }

  createFilmstrip(width: number, height: number): Promise<ImageBitmap> {
    return new Promise((resolve, _reject) => {
      const handleFilmstripResponse = (
        event: MessageEvent<FilmstripResponse>,
      ) => {
        if (event.data.action === 'filmstrip') {
          this.worker.removeEventListener('message', handleFilmstripResponse);
          resolve(event.data.filmstrip);
        }
      };

      this.worker.addEventListener('message', handleFilmstripResponse);

      this.sendRequest<FilmstripRequest>('filmstrip', {
        width,
        height,
      });
    });
  }

  setEffect(name: keyof Effects, index: EffectIndex, options?: EffectOptions) {
    this.sendRequest<SetEffectRequest>('setEffect', {
      name,
      index,
      options,
    });
  }

  encode(): void {
    this.sendRequest<EncodeVideoRequest>('encode');
  }

  initializeTracker(name: keyof Trackers, options: TrackerOptions): void {
    this.sendRequest<InitializeTrackerRequest>('initializeTracker', {
      name,
      options,
    });
  }

  startSession(videoUrl: string): Promise<string | null> {
    return new Promise(resolve => {
      const handleResponse = (
        event: MessageEvent<
          SessionStartedResponse | SessionStartFailedResponse
        >,
      ) => {
        if (event.data.action === 'sessionStarted') {
          this.worker.removeEventListener('message', handleResponse);
          resolve(event.data.sessionId);
        }
        if (event.data.action === 'sessionStartFailed') {
          this.worker.removeEventListener('message', handleResponse);
          resolve(null);
        }
      };

      this.worker.addEventListener('message', handleResponse);
      this.sendRequest<StartSessionRequest>('startSession', {
        videoUrl,
      });
    });
  }

  closeSession(): void {
    this.sendRequest<CloseSessionRequest>('closeSession');
  }

  logAnnotations(): void {
    this.sendRequest<LogAnnotationsRequest>('logAnnotations');
  }

  createTracklet(): Promise<BaseTracklet> {
    return new Promise(resolve => {
      const handleResponse = (event: MessageEvent<TrackletCreatedResponse>) => {
        if (event.data.action === 'trackletCreated') {
          this.worker.removeEventListener('message', handleResponse);
          resolve(event.data.tracklet);
        }
      };

      this.worker.addEventListener('message', handleResponse);

      this.sendRequest<CreateTrackletRequest>('createTracklet');
    });
  }

  deleteTracklet(trackletId: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const handleResponse = (event: MessageEvent<TrackletDeletedResponse>) => {
        if (event.data.action === 'trackletDeleted') {
          this.worker.removeEventListener('message', handleResponse);
          if (event.data.isSuccessful) {
            resolve();
          } else {
            reject(`could not delete tracklet ${trackletId}`);
          }
        }
      };
      this.worker.addEventListener('message', handleResponse);
      this.sendRequest<DeleteTrackletRequest>('deleteTracklet', {trackletId});
    });
  }

  updatePoints(
    objectId: number,
    points: SegmentationPoint[],
  ): Promise<boolean> {
    return new Promise(resolve => {
      const handleResponse = (event: MessageEvent<AddPointsResponse>) => {
        if (event.data.action === 'addPoints') {
          this.worker.removeEventListener('message', handleResponse);
          resolve(event.data.isSuccessful);
        }
      };

      this.worker.addEventListener('message', handleResponse);

      this.sendRequest<UpdatePointsRequest>('updatePoints', {
        frameIndex: this.frame,
        objectId,
        points,
      });
    });
  }

  clearPointsInFrame(objectId: number) {
    this.sendRequest<ClearPointsInFrameRequest>('clearPointsInFrame', {
      frameIndex: this.frame,
      objectId,
    });
  }

  clearPointsInVideo(): Promise<boolean> {
    return new Promise(resolve => {
      const handleResponse = (
        event: MessageEvent<ClearPointsInVideoResponse>,
      ) => {
        if (event.data.action === 'clearPointsInVideo') {
          this.worker.removeEventListener('message', handleResponse);
          resolve(event.data.isSuccessful);
        }
      };
      this.worker.addEventListener('message', handleResponse);
      this.sendRequest<ClearPointsInVideoRequest>('clearPointsInVideo');
    });
  }

  streamMasks(): void {
    this.sendRequest<StreamMasksRequest>('streamMasks', {
      frameIndex: this.frame,
    });
  }

  abortStreamMasks(): Promise<void> {
    return new Promise(resolve => {
      const handleAbortResponse = (
        event: MessageEvent<StreamingStateUpdateResponse>,
      ) => {
        if (
          event.data.action === 'streamingStateUpdate' &&
          event.data.state === 'aborted'
        ) {
          this.worker.removeEventListener('message', handleAbortResponse);
          resolve();
        }
      };

      this.worker.addEventListener('message', handleAbortResponse);
      this.sendRequest<AbortStreamMasksRequest>('abortStreamMasks');
    });
  }

  getWorker_ONLY_USE_WITH_CAUTION(): Worker {
    return this.worker;
  }

  /**
   * Convenient function to have typed postMessage.
   *
   * @param action Video worker action
   * @param message Actual payload
   * @param transfer Any object that should be transferred instead of cloned
   */
  protected sendRequest<T extends VideoWorkerRequest | TrackerRequest>(
    action: T['action'],
    payload?: Omit<T, 'action'>,
    transfer?: Transferable[],
  ) {
    this.worker.postMessage(
      {
        action,
        ...payload,
      },
      {
        transfer,
      },
    );
  }

  // // Override EventEmitter

  // addEventListener<K extends keyof WorkerEventMap>(
  //   type: K,
  //   listener: (ev: WorkerEventMap[K]) => unknown,
  // ): void {
  //   switch (type) {
  //     case 'frameUpdate':
  //       {
  //         const event: FrameUpdateEvent = {
  //           index: this.frameIndex,
  //         };
  //         // @ts-expect-error Incorrect typing. Not sure how to correctly type it
  //         listener(event);
  //       }
  //       break;
  //     case 'sessionStarted': {
  //       if (this.sessionId !== null) {
  //         const event: SessionStartedEvent = {
  //           sessionId: this.sessionId,
  //         };
  //         // @ts-expect-error Incorrect typing. Not sure how to correctly type it
  //         listener(event);
  //       }
  //     }
  //   }
  //   super.addEventListener(type, listener);
  // }
}
