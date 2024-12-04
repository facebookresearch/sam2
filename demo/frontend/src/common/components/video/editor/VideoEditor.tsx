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
import {VideoData} from '@/demo/atoms';
import stylex, {StyleXStyles} from '@stylexjs/stylex';
import {useSetAtom} from 'jotai';
import {PropsWithChildren, RefObject, useEffect, useRef} from 'react';
import Video, {VideoRef} from '../Video';
import {videoAtom} from './atoms';

const MAX_VIDEO_WIDTH = 1280;

const styles = stylex.create({
  editorContainer: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    borderRadius: '0.375rem',
    overflow: {
      default: 'clip',
      '@media screen and (max-width: 768px)': 'visible',
    },
  },
  videoContainer: {
    position: 'relative',
    flexGrow: 1,
    overflow: 'hidden',
    width: '100%',
    maxWidth: MAX_VIDEO_WIDTH,
  },
  layers: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    right: 0,
  },
  loadingMessage: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    padding: '6px 10px',
    backgroundColor: '#6441D2CC',
    color: '#FFF',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    borderRadius: '8px',
    fontSize: '0.8rem',
  },
});

export type InteractionLayerProps = {
  style: StyleXStyles;
  videoRef: RefObject<VideoRef>;
};

export type ControlsProps = {
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onPreviousFrame?: () => void;
  onNextFrame?: () => void;
};

type Props = PropsWithChildren<{
  video: VideoData;
  layers?: React.ReactNode;
  loading?: boolean;
}>;

export default function VideoEditor({
  video: inputVideo,
  layers,
  loading,
  children,
}: Props) {
  const videoRef = useRef<VideoRef>(null);
  const setVideo = useSetAtom(videoAtom);

  // Initialize video atom
  useEffect(() => {
    setVideo(videoRef.current);
    return () => {
      setVideo(null);
    };
  }, [setVideo]);

  return (
    <div {...stylex.props(styles.editorContainer)}>
      <div {...stylex.props(styles.videoContainer)}>
        <Video
          ref={videoRef}
          src={inputVideo.url}
          width={inputVideo.width}
          height={inputVideo.height}
          loading={loading}
        />
        <div {...stylex.props(styles.layers)}>{layers}</div>
      </div>
      {children}
    </div>
  );
}
