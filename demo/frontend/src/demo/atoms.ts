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
import {
  defaultMessageMap,
  MessagesEventMap,
} from '@/common/components/snackbar/DemoMessagesSnackbarUtils';
import {Effects} from '@/common/components/video/effects/Effects';
import {
  DemoEffect,
  highlightEffects,
} from '@/common/components/video/effects/EffectUtils';
import {
  BaseTracklet,
  SegmentationPoint,
  StreamingState,
} from '@/common/tracker/Tracker';
import type {DataArray} from '@/jscocotools/mask';
import {atom} from 'jotai';

export type VideoData = {
  path: string;
  posterPath: string | null | undefined;
  url: string;
  posterUrl: string;
  width: number;
  height: number;
};

export const frameIndexAtom = atom<number>(0);

export const inputVideoAtom = atom<VideoData | null>(null);

// #####################
// SESSION
// #####################

export type Session = {
  id: string;
  ranPropagation: boolean;
};

export const sessionAtom = atom<Session | null>(null);

// #####################
// STREAMING/PLAYBACK
// #####################

export const isVideoLoadingAtom = atom<boolean>(false);

export const streamingStateAtom = atom<StreamingState>('none');

export const isPlayingAtom = atom<boolean>(false);

export const isStreamingAtom = atom<boolean>(false);

// #####################
// OBJECTS
// #####################

export type TrackletMask = {
  mask: DataArray;
  isEmpty: boolean;
};

export type TrackletObject = {
  id: number;
  color: string;
  thumbnail: string | null;
  points: SegmentationPoint[][];
  masks: TrackletMask[];
  isInitialized: boolean;
};

const MAX_NUMBER_TRACKLET_OBJECTS = 3;

export const activeTrackletObjectIdAtom = atom<number | null>(0);

export const activeTrackletObjectAtom = atom<BaseTracklet | null>(get => {
  const objectId = get(activeTrackletObjectIdAtom);
  const tracklets = get(trackletObjectsAtom);
  return tracklets.find(obj => obj.id === objectId) ?? null;
});

export const trackletObjectsAtom = atom<BaseTracklet[]>([]);

export const maxTrackletObjectIdAtom = atom<number>(get => {
  const tracklets = get(trackletObjectsAtom);
  return tracklets.reduce((prev, curr) => Math.max(prev, curr.id), 0);
});

export const isTrackletObjectLimitReachedAtom = atom<boolean>(
  get => get(trackletObjectsAtom).length >= MAX_NUMBER_TRACKLET_OBJECTS,
);

export const areTrackletObjectsInitializedAtom = atom<boolean>(get =>
  get(trackletObjectsAtom).every(obj => obj.isInitialized),
);

export const isFirstClickMadeAtom = atom(get => {
  const tracklets = get(trackletObjectsAtom);
  return tracklets.some(tracklet => tracklet.points.length > 0);
});

export const pointsAtom = atom<SegmentationPoint[]>(get => {
  const frameIndex = get(frameIndexAtom);
  const activeTracklet = get(activeTrackletObjectAtom);
  return activeTracklet?.points[frameIndex] ?? [];
});

export const labelTypeAtom = atom<'positive' | 'negative'>('positive');

export const isAddObjectEnabledAtom = atom<boolean>(get => {
  const session = get(sessionAtom);
  const trackletsInitialized = get(areTrackletObjectsInitializedAtom);
  const isObjectLimitReached = get(isTrackletObjectLimitReachedAtom);
  return (
    session?.ranPropagation === false &&
    trackletsInitialized &&
    !isObjectLimitReached
  );
});

export const codeEditorOpenedAtom = atom<boolean>(false);

export const tutorialVideoEnabledAtom = atom<boolean>(true);

// #####################
// Effects
// #####################

type EffectConfig = {
  name: keyof Effects;
  variant: number;
  numVariants: number;
};

export const activeBackgroundEffectAtom = atom<EffectConfig>({
  name: 'Original',
  variant: 0,
  numVariants: 0,
});

export const activeHighlightEffectAtom = atom<EffectConfig>({
  name: 'Overlay',
  variant: 0,
  numVariants: 0,
});

export const activeHighlightEffectGroupAtom =
  atom<DemoEffect[]>(highlightEffects);

// #####################
// Toolbar
// #####################

export const toolbarTabIndex = atom<number>(0);

// #####################
// Messages snackbar
// #####################

export const messageMapAtom = atom<MessagesEventMap>(defaultMessageMap);

// #####################
// Upload state
// #####################

export const uploadingStateAtom = atom<'default' | 'uploading' | 'error'>(
  'default',
);
