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
import {BaseTracklet, SegmentationPoint} from '@/common/tracker/Tracker';
import {TrackerOptions, Trackers} from '@/common/tracker/Trackers';
import {PauseFilled, PlayFilledAlt} from '@carbon/icons-react';
import stylex, {StyleXStyles} from '@stylexjs/stylex';
import {
  CSSProperties,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from 'react';
import {Button} from 'react-daisyui';

import {EffectIndex, Effects} from '@/common/components/video/effects/Effects';
import useReportError from '@/common/error/useReportError';
import Logger from '@/common/logger/Logger';
import {isPlayingAtom, isVideoLoadingAtom} from '@/demo/atoms';
import {color} from '@/theme/tokens.stylex';
import {useAtom} from 'jotai';
import useResizeObserver from 'use-resize-observer';
import VideoLoadingOverlay from './VideoLoadingOverlay';
import {
  StreamingStateUpdateEvent,
  VideoWorkerEventMap,
} from './VideoWorkerBridge';
import {EffectOptions} from './effects/Effect';
import useVideoWorker from './useVideoWorker';

const styles = stylex.create({
  container: {
    position: 'relative',
    width: '100%',
    height: '100%',
  },
  canvasContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: color['gray-800'],
    width: '100%',
    height: '100%',
  },
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    padding: 8,
    background: 'linear-gradient(#00000000, #000000ff)',
  },
  controlButton: {
    color: 'white',
  },
});

type Props = {
  src: string;
  width: number;
  height: number;
  loading?: boolean;
  containerStyle?: StyleXStyles<{
    position: CSSProperties['position'];
  }>;
  canvasStyle?: StyleXStyles<{
    width: CSSProperties['width'];
  }>;
  controls?: boolean;
  createVideoWorker?: () => Worker;
};

export type VideoRef = {
  getCanvas(): HTMLCanvasElement | null;
  get width(): number;
  get height(): number;
  get frame(): number;
  set frame(index: number);
  get numberOfFrames(): number;
  play(): void;
  pause(): void;
  stop(): void;
  previousFrame(): void;
  nextFrame(): void;
  setEffect(
    name: keyof Effects,
    index: EffectIndex,
    options?: EffectOptions,
  ): void;
  encode(): void;
  streamMasks(): void;
  abortStreamMasks(): Promise<void>;
  addEventListener<K extends keyof VideoWorkerEventMap>(
    type: K,
    listener: (ev: VideoWorkerEventMap[K]) => unknown,
  ): void;
  removeEventListener<K extends keyof VideoWorkerEventMap>(
    type: K,
    listener: (ev: VideoWorkerEventMap[K]) => unknown,
  ): void;
  createFilmstrip(width: number, height: number): Promise<ImageBitmap>;
  // Tracker
  initializeTracker(name: keyof Trackers, options?: TrackerOptions): void;
  startSession(videoUrl: string): Promise<string | null>;
  closeSession(): void;
  logAnnotations(): void;
  createTracklet(): Promise<BaseTracklet>;
  deleteTracklet(trackletId: number): Promise<void>;
  updatePoints(trackletId: number, points: SegmentationPoint[]): void;
  clearPointsInVideo(): Promise<boolean>;
  getWorker_ONLY_USE_WITH_CAUTION(): Worker;
};

export default forwardRef<VideoRef, Props>(function Video(
  {
    src,
    width,
    height,
    containerStyle,
    canvasStyle,
    createVideoWorker,
    controls = false,
    loading = false,
  },
  ref,
) {
  const reportError = useReportError();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useAtom(isPlayingAtom);
  const [isVideoLoading, setIsVideoLoading] = useAtom(isVideoLoadingAtom);

  const bridge = useVideoWorker(src, canvasRef, {
    createVideoWorker,
  });

  const {
    ref: resizeObserverRef,
    width: resizeWidth = 1,
    height: resizeHeight = 1,
  } = useResizeObserver<HTMLDivElement>();

  const canvasHeight = useMemo(() => {
    const resizeRatio = resizeWidth / width;
    return Math.min(height * resizeRatio, resizeHeight);
  }, [resizeWidth, height, width, resizeHeight]);

  useImperativeHandle(
    ref,
    () => ({
      getCanvas() {
        return canvasRef.current;
      },
      get width() {
        return bridge.width;
      },
      get height() {
        return bridge.width;
      },
      get frame() {
        return bridge.frame;
      },
      set frame(index: number) {
        bridge.frame = index;
      },
      get numberOfFrames() {
        return bridge.numberOfFrames;
      },
      play(): void {
        bridge.play();
      },
      pause(): void {
        bridge.pause();
      },
      stop(): void {
        bridge.stop();
      },
      previousFrame(): void {
        bridge.previousFrame();
      },
      nextFrame(): void {
        bridge.nextFrame();
      },
      setEffect(
        name: keyof Effects,
        index: number,
        options?: EffectOptions,
      ): void {
        bridge.setEffect(name, index, options);
      },
      encode(): void {
        bridge.encode();
      },
      streamMasks(): void {
        bridge.streamMasks();
      },
      abortStreamMasks(): Promise<void> {
        return bridge.abortStreamMasks();
      },
      addEventListener<K extends keyof VideoWorkerEventMap>(
        type: K,
        listener: (ev: VideoWorkerEventMap[K]) => unknown,
      ): void {
        bridge.addEventListener(type, listener);
      },
      removeEventListener<K extends keyof VideoWorkerEventMap>(
        type: K,
        listener: (ev: VideoWorkerEventMap[K]) => unknown,
      ): void {
        bridge.removeEventListener(type, listener);
      },
      createFilmstrip(width: number, height: number): Promise<ImageBitmap> {
        return bridge.createFilmstrip(width, height);
      },
      // Tracker
      initializeTracker(name: keyof Trackers, options: TrackerOptions): void {
        bridge.initializeTracker(name, options);
      },
      startSession(videoUrl: string): Promise<string | null> {
        return bridge.startSession(videoUrl);
      },
      closeSession(): void {
        bridge.closeSession();
      },
      logAnnotations(): void {
        bridge.logAnnotations();
      },
      createTracklet(): Promise<BaseTracklet> {
        return bridge.createTracklet();
      },
      deleteTracklet(trackletId: number): Promise<void> {
        return bridge.deleteTracklet(trackletId);
      },
      updatePoints(trackletId: number, points: SegmentationPoint[]): void {
        bridge.updatePoints(trackletId, points);
      },
      clearPointsInVideo(): Promise<boolean> {
        return bridge.clearPointsInVideo();
      },
      getWorker_ONLY_USE_WITH_CAUTION() {
        return bridge.getWorker_ONLY_USE_WITH_CAUTION();
      },
    }),
    [bridge],
  );

  // Handle video playback events (get playback state to main thread)
  useEffect(() => {
    let isPlaying = false;

    function onFocus() {
      // Workaround for Safari where the video frame renders black on
      // unknown events. Trigger re-render frame on focus.
      if (!isPlaying) {
        bridge.goToFrame(bridge.frame);
      }
    }

    function onVisibilityChange() {
      // Workaround for Safari where the video frame renders black on
      // visibility change hidden. Returning to visible shows a black
      // frame instead of rendering the current frame.
      if (document.visibilityState === 'visible' && !isPlaying) {
        bridge.goToFrame(bridge.frame);
      }
    }

    function onError(event: ErrorEvent) {
      const error = event.error;
      Logger.error(error);
      reportError(error);
    }

    function onPlay() {
      isPlaying = true;
      setIsPlaying(true);
    }
    function onPause() {
      isPlaying = false;
      setIsPlaying(false);
    }

    function onStreamingDone(event: StreamingStateUpdateEvent) {
      // continue to play after streaming is done (state is "full")
      if (event.state === 'full') {
        bridge.play();
      }
    }

    function onLoadStart() {
      setIsVideoLoading(true);
    }

    function onDecodeStart() {
      setIsVideoLoading(false);
    }

    window.addEventListener('focus', onFocus);
    window.addEventListener('visibilitychange', onVisibilityChange);
    bridge.addEventListener('error', onError);
    bridge.addEventListener('play', onPlay);
    bridge.addEventListener('pause', onPause);
    bridge.addEventListener('streamingStateUpdate', onStreamingDone);
    bridge.addEventListener('loadstart', onLoadStart);
    bridge.addEventListener('decode', onDecodeStart);
    return () => {
      window.removeEventListener('focus', onFocus);
      window.removeEventListener('visibilitychange', onVisibilityChange);
      bridge.removeEventListener('error', onError);
      bridge.removeEventListener('play', onPlay);
      bridge.removeEventListener('pause', onPause);
      bridge.removeEventListener('streamingStateUpdate', onStreamingDone);
      bridge.removeEventListener('loadstart', onLoadStart);
      bridge.removeEventListener('decode', onDecodeStart);
    };
  }, [bridge, reportError, setIsPlaying, setIsVideoLoading]);

  return (
    <div
      {...stylex.props(containerStyle ?? styles.container)}
      ref={resizeObserverRef}>
      <div {...stylex.props(styles.canvasContainer)}>
        {(isVideoLoading || loading) && <VideoLoadingOverlay />}
        <canvas
          ref={canvasRef}
          {...stylex.props(canvasStyle)}
          className="lg:rounded-[4px]"
          width={width}
          height={height}
          style={{
            height: canvasHeight,
          }}
        />
      </div>
      {controls && (
        <div {...stylex.props(styles.controls)}>
          <Button
            color="ghost"
            size="xs"
            startIcon={
              isPlaying ? (
                <PauseFilled
                  {...stylex.props(styles.controlButton)}
                  size={14}
                />
              ) : (
                <PlayFilledAlt
                  {...stylex.props(styles.controlButton)}
                  size={14}
                />
              )
            }
            onClick={() => {
              isPlaying ? bridge.pause() : bridge.play();
            }}
          />
        </div>
      )}
    </div>
  );
});
