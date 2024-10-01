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
import stylex from '@stylexjs/stylex';
import {useMemo} from 'react';
import useResizeObserver from 'use-resize-observer';
import useVideo from '../editor/useVideo';

const styles = stylex.create({
  container: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
  },
});

type Props = {
  points: SegmentationPoint[];
  onRemovePoint: (point: SegmentationPoint) => void;
};

export function PointsLayer({points, onRemovePoint}: Props) {
  const video = useVideo();
  const videoCanvas = useMemo(() => video?.getCanvas(), [video]);

  const {
    ref,
    width: containerWidth = 1,
    height: containerHeight = 1,
  } = useResizeObserver<SVGElement>();

  const canvasWidth = videoCanvas?.width ?? 1;
  const canvasHeight = videoCanvas?.height ?? 1;

  const sizeMultiplier = useMemo(() => {
    const widthMultiplier = canvasWidth / containerWidth;
    const heightMultiplier = canvasHeight / containerHeight;

    return Math.max(widthMultiplier, heightMultiplier);
  }, [canvasWidth, canvasHeight, containerWidth, containerHeight]);

  const pointRadius = useMemo(() => 8 * sizeMultiplier, [sizeMultiplier]);
  const pointStroke = useMemo(() => 2 * sizeMultiplier, [sizeMultiplier]);

  return (
    <svg
      ref={ref}
      {...stylex.props(styles.container)}
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}>
      {/*
       * This is a debug element to verify the SVG element overlays
       * perfectly with the canvas element.
       */}
      {/*
      <rect
        fill="rgba(255, 255, 0, 0.5)"
        width={decodedVideo?.width}
        height={decodedVideo?.height}
      />
      */}
      {/* Render points */}
      {points.map((point, idx) => {
        const isAdd = point[2] === 1;
        return (
          <g key={idx} className="cursor-pointer">
            <circle
              className="stroke-white hover:stroke-gray-400"
              pointerEvents="visiblePainted"
              cx={point[0]}
              cy={point[1]}
              r={pointRadius}
              fill={isAdd ? '#000000' : '#E6193B'}
              strokeWidth={pointStroke}
              onClick={event => {
                event.stopPropagation();
                onRemovePoint(point);
              }}
            />
            <line
              x1={point[0] - pointRadius / 2}
              y1={point[1]}
              x2={point[0] + pointRadius / 2}
              y2={point[1]}
              strokeWidth={pointStroke}
              stroke="white"
            />
            {isAdd && (
              <line
                x1={point[0]}
                y1={point[1] - pointRadius / 2}
                x2={point[0]}
                y2={point[1] + pointRadius / 2}
                strokeWidth={pointStroke}
                stroke="white"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
