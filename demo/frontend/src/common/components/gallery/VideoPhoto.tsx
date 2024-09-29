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
import Logger from '@/common/logger/Logger';
import stylex from '@stylexjs/stylex';
import {
  CSSProperties,
  MouseEventHandler,
  useCallback,
  useEffect,
  useRef,
} from 'react';

const styles = stylex.create({
  background: {
    backgroundRepeat: 'no-repeat',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    cursor: 'pointer',
  },
  video: {
    width: '100%',
    height: '100%',
  },
});

type Props = {
  onClick: MouseEventHandler<HTMLVideoElement> | undefined;
  src: string;
  poster: string;
  style: CSSProperties;
};

export default function VideoPhoto({src, poster, style, onClick}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const playPromiseRef = useRef<Promise<void> | null>(null);

  const play = useCallback(() => {
    const video = videoRef.current;
    // Only play video if it is not already playing
    if (video != null && video.paused) {
      // This quirky way of handling video play/pause in the browser is needed
      // due to the async nature of the video play API:
      // https://developer.chrome.com/blog/play-request-was-interrupted/
      const playPromise = video.play();
      playPromise.catch(error => {
        Logger.error('Failed to play video', error);
      });
      playPromiseRef.current = playPromise;
    }
  }, []);

  const pause = useCallback(() => {
    // Only pause video if it is playing
    const playPromise = playPromiseRef.current;
    if (playPromise != null) {
      playPromise
        .then(() => {
          videoRef.current?.pause();
        })
        .catch(error => {
          Logger.error('Failed to pause video', error);
        })
        .finally(() => {
          playPromiseRef.current = null;
        });
    }
  }, []);

  useEffect(() => {
    return () => {
      pause();
    };
  }, [pause]);

  return (
    <div
      style={{
        ...style,
        backgroundImage: `url(${poster})`,
      }}
      {...stylex.props(styles.background)}>
      <video
        ref={videoRef}
        {...stylex.props(styles.video)}
        preload="none"
        playsInline
        loop
        muted
        title="Gallery Video"
        poster={poster}
        onMouseEnter={play}
        onMouseLeave={pause}
        onClick={onClick}>
        <source src={src} type="video/mp4" />
        Sorry, your browser does not support embedded videos.
      </video>
    </div>
  );
}
