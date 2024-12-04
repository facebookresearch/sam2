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
import SelectedFrameHelper from '@/common/components/video/filmstrip/SelectedFrameHelper';
import {isPlayingAtom} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtomValue, useSetAtom} from 'jotai';
import {CanvasSpace, Pt} from 'pts';
import {useCallback, useEffect, useMemo, useRef} from 'react';
import {PtsCanvas, PtsCanvasImperative} from 'react-pts-canvas';
import {VideoRef} from '../Video';
import {DecodeEvent, FrameUpdateEvent} from '../VideoWorkerBridge';
import useVideo from '../editor/useVideo';
import {
  drawFilmstrip,
  drawMarker,
  getPointerPosition,
  getTimeFromFrame,
} from './FilmstripUtil';
import {selectedFrameHelperAtom} from './atoms';
import useDisableScrolling from './useDisableScrolling';

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
  },
  filmstripWrapper: {
    position: 'relative',
    width: '100%',
    height: '5rem' /* 80px */,
  },
  filmstrip: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    right: 0,
    cursor: 'col-resize',
    overflow: 'hidden',
  },
  canvas: {
    width: '100%',
    height: '100%',
  },
});

export const PADDING_TOP = 30;
export const PADDING_BOTTOM = 0;

export default function VideoFilmstrip() {
  const video = useVideo();
  const ptsCanvasRef = useRef<PtsCanvasImperative | null>(null);
  const filmstripRef = useRef<ImageBitmap | null>(null);
  const isPlayingOnPointerDownRef = useRef<boolean>(false);
  const isPlaying = useAtomValue(isPlayingAtom);

  const {enable: enableScrolling, disable: disableScrolling} =
    useDisableScrolling();

  const pointerPositionRef = useRef<Pt | null>(null);
  const animateRAFHandle = useRef<number | null>(null);

  const selectedFrameHelper = useMemo(() => new SelectedFrameHelper(1, 1), []);
  const setSelectedFrameHelper = useSetAtom(selectedFrameHelperAtom);

  const fpsRef = useRef<number>(30);

  useEffect(() => {
    function onDecode(event: DecodeEvent) {
      video?.removeEventListener('decode', onDecode);
      fpsRef.current = event.fps;
    }
    video?.addEventListener('decode', onDecode);
    return () => {
      video?.removeEventListener('decode', onDecode);
    };
  }, [video]);

  useEffect(() => {
    setSelectedFrameHelper(selectedFrameHelper);
  }, [setSelectedFrameHelper, selectedFrameHelper]);

  const computeFrame = useCallback(
    (normalizedPosition: number): {index: number} | null => {
      if (video == null) {
        return null;
      }

      const numFrames = video.numberOfFrames;
      const index = Math.min(
        Math.max(0, Math.floor(normalizedPosition * numFrames)),
        numFrames - 1,
      );
      // The frame is needed for the CAE model. Do we still want to support it?
      // return {image: decodedVideo.frames[index], index: index};
      return {index};
    },
    [video],
  );

  const createFilmstrip = useCallback(
    async (
      video: VideoRef | null,
      space: CanvasSpace | undefined,
      frameIndex?: number,
    ) => {
      if (video === null || space == undefined) {
        return;
      }

      const bitmap: ImageBitmap = await video?.createFilmstrip(
        space.width,
        space.height - (PADDING_TOP - PADDING_BOTTOM),
      );

      filmstripRef.current = bitmap;

      selectedFrameHelper.reset(video.numberOfFrames, space.width, frameIndex); // also reset index to first frame

      return bitmap;
    },
    [selectedFrameHelper],
  );

  // Custom animation handler
  const handleRAF = useCallback(() => {
    animateRAFHandle.current = null;
    const space = ptsCanvasRef.current?.getSpace();
    const form = ptsCanvasRef.current?.getForm();
    if (space == undefined || form == undefined) {
      return;
    }

    // Clear space, in particular clearing the frame index number of
    // previous renders.
    space.clear();

    drawFilmstrip(filmstripRef.current, space, form);

    const scanLabel =
      selectedFrameHelper.isScanning &&
      pointerPositionRef.current !== null &&
      fpsRef.current !== null &&
      getTimeFromFrame(
        computeFrame(pointerPositionRef.current.x / space.width)?.index ?? 0,
        fpsRef.current,
      );

    drawMarker(
      space,
      form,
      selectedFrameHelper,
      pointerPositionRef.current,
      scanLabel,
      fpsRef.current,
    );
  }, [computeFrame, selectedFrameHelper]);

  const handleAnimate = useCallback(() => {
    if (animateRAFHandle.current === null) {
      animateRAFHandle.current = requestAnimationFrame(handleRAF);
    }
  }, [handleRAF]);

  const handleFrameUpdate = useCallback(
    (event: FrameUpdateEvent) => {
      if (!selectedFrameHelper.isScanning) {
        selectedFrameHelper.select(event.index);
      }
      handleAnimate();
    },
    [handleAnimate, selectedFrameHelper],
  );

  // Register a frame update listener on the video to update the filmstrip
  // indicator when the video changes frames.
  useEffect(() => {
    video?.addEventListener('frameUpdate', handleFrameUpdate);
    return () => {
      video?.removeEventListener('frameUpdate', handleFrameUpdate);
    };
  }, [video, handleFrameUpdate]);

  // Initiate filmstrip image
  useEffect(() => {
    const space = ptsCanvasRef.current?.getSpace();

    async function onLoadStart() {
      await createFilmstrip(video, space, 0);
      handleAnimate();
    }

    async function progress() {
      await createFilmstrip(video, space, 0);
      handleAnimate();
    }

    void progress();

    video?.addEventListener('loadstart', onLoadStart);
    video?.addEventListener('decode', progress);

    return () => {
      video?.removeEventListener('loadstart', onLoadStart);
      video?.removeEventListener('decode', progress);
    };
  }, [createFilmstrip, selectedFrameHelper, handleAnimate, video]);

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.filmstripWrapper)}>
        <div {...stylex.props(styles.filmstrip)}>
          <PtsCanvas
            {...stylex.props(styles.canvas)}
            ref={ptsCanvasRef}
            background="transparent"
            resize={true}
            refresh={false}
            play={false}
            onPtsResize={async space => {
              if (video != null && space != undefined) {
                selectedFrameHelper.reset(video.numberOfFrames, space.width);
              }
              if (video !== null) {
                await createFilmstrip(video, space);
              }
              handleAnimate();
            }}
            onPointerDown={event => {
              const canvas = ptsCanvasRef.current?.getCanvas();
              canvas?.setPointerCapture(event.pointerId);

              // Disable page scrolling while interacting with the filmstrip
              disableScrolling();

              pointerPositionRef.current = getPointerPosition(event);
              selectedFrameHelper.scan(true);

              // Pause the video when a user initially has their pointer down.
              // Playback will resume once the onPointerUp event is triggered.
              isPlayingOnPointerDownRef.current = isPlaying;
              if (isPlaying) {
                video?.pause();
              }
            }}
            onPointerUp={event => {
              // Enable page scrolling after interaction with filmstrip is done
              enableScrolling();

              const space = ptsCanvasRef.current?.getSpace();
              if (space != undefined) {
                pointerPositionRef.current = getPointerPosition(event);

                selectedFrameHelper.scan(false);
                const frame = computeFrame(
                  pointerPositionRef.current.x / space.size.x,
                );
                if (
                  frame != null &&
                  selectedFrameHelper.index !== frame.index
                ) {
                  selectedFrameHelper.select(frame.index);
                  if (video !== null) {
                    video.frame = frame.index;
                    if (isPlayingOnPointerDownRef.current) {
                      video.play();
                    }
                  }
                }
                handleAnimate();
              }

              pointerPositionRef.current = null;
            }}
            onPointerMove={event => {
              if (
                !selectedFrameHelper.isScanning ||
                pointerPositionRef.current === null
              ) {
                return;
              }

              const space = ptsCanvasRef.current?.getSpace();
              const form = ptsCanvasRef.current?.getForm();
              if (
                selectedFrameHelper.isScanning &&
                space != null &&
                form != null
              ) {
                pointerPositionRef.current = getPointerPosition(event);

                const frame = computeFrame(
                  pointerPositionRef.current.x / space.size.x,
                );
                if (frame != null) {
                  handleAnimate();
                  if (video !== null) {
                    video.frame = frame.index;
                  }
                }
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}
