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
import ClearAllPointsInVideoButton from '@/common/components/annotations/ClearAllPointsInVideoButton';
import ObjectThumbnail from '@/common/components/annotations/ObjectThumbnail';
import {OBJECT_TOOLBAR_INDEX} from '@/common/components/toolbar/ToolbarConfig';
import {BaseTracklet} from '@/common/tracker/Tracker';
import {activeTrackletObjectIdAtom, trackletObjectsAtom} from '@/demo/atoms';
import {spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useAtomValue, useSetAtom} from 'jotai';

const styles = stylex.create({
  container: {
    display: 'flex',
    padding: spacing[5],
    borderTop: '1px solid #DEE3E9',
  },
  trackletsContainer: {
    flexGrow: 1,
    display: 'flex',
    alignItems: 'center',
    gap: spacing[5],
  },
});

type Props = {
  showActiveObject: () => void;
  onTabChange: (newIndex: number) => void;
};

export default function MobileObjectsList({
  showActiveObject,
  onTabChange,
}: Props) {
  const tracklets = useAtomValue(trackletObjectsAtom);
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);

  function handleSelectObject(tracklet: BaseTracklet) {
    setActiveTrackletId(tracklet.id);
    showActiveObject();
  }

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.trackletsContainer)}>
        {tracklets.map(tracklet => {
          const {id, color, thumbnail} = tracklet;
          return (
            <ObjectThumbnail
              key={id}
              color={color}
              thumbnail={thumbnail}
              onClick={() => {
                handleSelectObject(tracklet);
              }}
            />
          );
        })}
      </div>
      <ClearAllPointsInVideoButton
        onRestart={() => onTabChange(OBJECT_TOOLBAR_INDEX)}
      />
    </div>
  );
}
