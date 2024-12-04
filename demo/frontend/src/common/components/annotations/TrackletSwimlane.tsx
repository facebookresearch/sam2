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
import useSelectedFrameHelper from '@/common/components/video/filmstrip/useSelectedFrameHelper';
import {BaseTracklet, DatalessMask} from '@/common/tracker/Tracker';
import {spacing, w} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useMemo} from 'react';

const styles = stylex.create({
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: spacing[4],
    width: '100%',
  },
  trackletNameContainer: {
    width: w[12],
    textAlign: 'center',
    fontSize: '10px',
    color: 'white',
  },
  swimlaneContainer: {
    flexGrow: 1,
    position: 'relative',
    display: 'flex',
    height: 12,
    marginVertical: '0.25rem' /* 4px */,
    '@media screen and (max-width: 768px)': {
      marginVertical: 0,
    },
  },
  swimlane: {
    position: 'absolute',
    left: 0,
    top: '50%',
    width: '100%',
    height: 1,
    transform: 'translate3d(0, -50%, 0)',
    opacity: 0.4,
  },
  segment: {
    position: 'absolute',
    top: '50%',
    height: 1,
    transform: 'translate3d(0, -50%, 0)',
  },
  segmentationPoint: {
    position: 'absolute',
    top: '50%',
    transform: 'translate3d(0, -50%, 0)',
    borderRadius: '50%',
    cursor: 'pointer',
    width: 12,
    height: 12,
    '@media screen and (max-width: 768px)': {
      width: 8,
      height: 8,
    },
  },
});

type SwimlineSegment = {
  left: number;
  width: number;
};

type Props = {
  tracklet: BaseTracklet;
  onSelectFrame: (tracklet: BaseTracklet, index: number) => void;
};

function getSwimlaneSegments(masks: DatalessMask[]): SwimlineSegment[] {
  if (masks.length === 0) {
    return [];
  }

  const swimlineSegments: SwimlineSegment[] = [];
  let left = -1;

  for (let frameIndex = 0; frameIndex < masks.length; ++frameIndex) {
    const isEmpty = masks?.[frameIndex]?.isEmpty ?? true;
    if (left === -1 && !isEmpty) {
      left = frameIndex;
    } else if (left !== -1 && (isEmpty || frameIndex == masks.length - 1)) {
      swimlineSegments.push({
        left,
        width: frameIndex - left + 1,
      });
      left = -1;
    }
  }

  return swimlineSegments;
}

export default function TrackletSwimlane({tracklet, onSelectFrame}: Props) {
  const selection = useSelectedFrameHelper();

  const segments = useMemo(() => {
    return getSwimlaneSegments(tracklet.masks);
  }, [tracklet.masks]);

  const framesWithPoints = tracklet.points.reduce<number[]>(
    (frames, pts, frameIndex) => {
      if (pts != null && pts.length > 0) {
        frames.push(frameIndex);
      }
      return frames;
    },
    [],
  );

  if (selection === null) {
    return;
  }

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.trackletNameContainer)}>
        Object {tracklet.id + 1}
      </div>
      <div {...stylex.props(styles.swimlaneContainer)}>
        <div
          {...stylex.props(styles.swimlane)}
          style={{
            backgroundColor: tracklet.color,
          }}
        />
        {segments.map(segment => {
          return (
            <div
              key={segment.left}
              {...stylex.props(styles.segment)}
              style={{
                backgroundColor: tracklet.color,
                left: selection.toPosition(segment.left),
                width: selection.toPosition(segment.width),
              }}
            />
          );
        })}
        {framesWithPoints.map(index => {
          return (
            <div
              key={`frame${index}`}
              onClick={() => {
                onSelectFrame?.(tracklet, index);
              }}
              {...stylex.props(styles.segmentationPoint)}
              style={{
                left: Math.floor(selection.toPosition(index) - 4),
                backgroundColor: tracklet.color,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
