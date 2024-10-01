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
import React from 'react';

export type VideoAspectRatio = 'wide' | 'square' | 'normal' | 'fill';

export type VideoProps = {
  src: string;
  aspectRatio?: VideoAspectRatio;
  className?: string;
  containerClassName?: string;
} & React.VideoHTMLAttributes<HTMLVideoElement>;

export default function StaticVideoPlayer({
  src,
  aspectRatio,
  className = '',
  containerClassName = '',
  ...props
}: VideoProps) {
  let aspect =
    aspectRatio === 'wide'
      ? `aspect-video`
      : aspectRatio === 'square'
        ? 'aspect-square'
        : 'aspect-auto';

  let videoSize = '';

  if (aspectRatio === 'fill') {
    aspect =
      'absolute object-cover right-0 bottom-0 min-w-full min-h-full h-full';
    videoSize = 'w-full h-full object-cover object-center';
  }

  return (
    <div
      className={`w-full relative flex flex-col ${aspect} ${containerClassName}`}>
      <video className={`m-0 ${videoSize} ${className}`} {...props}>
        <source src={src} type="video/mp4" />
        Sorry, your browser does not support embedded videos.
      </video>
    </div>
  );
}
