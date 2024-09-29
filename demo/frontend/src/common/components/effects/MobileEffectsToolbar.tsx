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
import EffectsCarousel from '@/common/components/effects/EffectsCarousel';
import {backgroundEffects} from '@/common/components/effects/EffectsUtils';
import useVideoEffect from '@/common/components/video/editor/useVideoEffect';
import {
  EffectIndex,
  effectPresets,
} from '@/common/components/video/effects/Effects';
import {ListBoxes, MagicWand, MagicWandFilled} from '@carbon/icons-react';
import {useCallback, useRef, useState} from 'react';
import {Button} from 'react-daisyui';

import EffectsToolbarBottomActions from '@/common/components/effects/EffectsToolbarBottomActions';
import ToolbarProgressChip from '@/common/components/toolbar/ToolbarProgressChip';
import {
  activeBackgroundEffectAtom,
  activeHighlightEffectAtom,
  activeHighlightEffectGroupAtom,
} from '@/demo/atoms';
import {BLUE_PINK_FILL} from '@/theme/gradientStyle';
import {useAtomValue} from 'jotai';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function MobileEffectsToolbar({onTabChange}: Props) {
  const preset = useRef(0);
  const setEffect = useVideoEffect();
  const [showEffectsCarousels, setShowEffectsCarousels] = useState<boolean>();

  const activeBackground = useAtomValue(activeBackgroundEffectAtom);
  const activeHighlight = useAtomValue(activeHighlightEffectAtom);
  const activeHighlightEffectsGroup = useAtomValue(
    activeHighlightEffectGroupAtom,
  );

  const handleTogglePreset = useCallback(() => {
    preset.current++;
    const [background, highlight] =
      effectPresets[preset.current % effectPresets.length];
    setEffect(background.name, EffectIndex.BACKGROUND, {
      variant: background.variant,
    });
    setEffect(highlight.name, EffectIndex.HIGHLIGHT, {
      variant: highlight.variant,
    });
  }, [setEffect]);

  return (
    <div className="w-full">
      {showEffectsCarousels ? (
        <div className="flex gap-2 px-2 py-4 items-center p-6">
          <Button
            color="ghost"
            className="mt-6 !px-2 !text-[#FB73A5]"
            startIcon={<MagicWand size={20} />}
            onClick={handleTogglePreset}
          />
          <EffectsCarousel
            label="Highlights"
            effects={activeHighlightEffectsGroup}
            activeEffect={activeHighlight.name}
            index={1}
          />
          <EffectsCarousel
            label="Background"
            effects={backgroundEffects}
            activeEffect={activeBackground.name}
            index={0}
          />
        </div>
      ) : (
        <div className="flex flex-col gap-6 p-6">
          <div className="text-sm text-white">
            <ToolbarProgressChip />
            Apply visual effects to your selected objects and the background.
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Button
              color="ghost"
              endIcon={<MagicWandFilled size={20} />}
              className={`font-bold bg-black !rounded-full !bg-gradient-to-br ${BLUE_PINK_FILL} border-none text-white`}
              onClick={handleTogglePreset}>
              Surprise Me
            </Button>
            <Button
              color="ghost"
              className={`font-bold bg-black !rounded-full border-none text-white`}
              startIcon={<ListBoxes size={20} />}
              onClick={() => setShowEffectsCarousels(true)}>
              More effects
            </Button>
          </div>
        </div>
      )}

      <EffectsToolbarBottomActions onTabChange={onTabChange} />
    </div>
  );
}
