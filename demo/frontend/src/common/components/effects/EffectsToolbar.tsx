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
import BackgroundEffects from '@/common/components/effects/BackgroundEffects';
import EffectsToolbarBottomActions from '@/common/components/effects/EffectsToolbarBottomActions';
import EffectsToolbarHeader from '@/common/components/effects/EffectsToolbarHeader';
import HighlightEffects from '@/common/components/effects/HighlightEffects';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import {useEffect, useRef} from 'react';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function EffectsToolbar({onTabChange}: Props) {
  const isEffectsMessageShown = useRef(false);
  const {enqueueMessage} = useMessagesSnackbar();

  useEffect(() => {
    if (!isEffectsMessageShown.current) {
      isEffectsMessageShown.current = true;
      enqueueMessage('effectsMessage');
    }
  }, [enqueueMessage]);

  return (
    <div className="flex flex-col h-full">
      <EffectsToolbarHeader />
      <div className="grow overflow-y-auto">
        <HighlightEffects />
        <BackgroundEffects />
      </div>
      <EffectsToolbarBottomActions onTabChange={onTabChange} />
    </div>
  );
}
