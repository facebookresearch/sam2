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
import ToolbarHeaderWrapper from '@/common/components/toolbar/ToolbarHeaderWrapper';
import useVideoEffect from '@/common/components/video/editor/useVideoEffect';
import {
  EffectIndex,
  effectPresets,
} from '@/common/components/video/effects/Effects';
import {BLUE_PINK_FILL} from '@/theme/gradientStyle';
import {MagicWandFilled} from '@carbon/icons-react';
import {useCallback, useRef} from 'react';
import {Button} from 'react-daisyui';

export default function EffectsToolbarHeader() {
  const preset = useRef(0);
  const setEffect = useVideoEffect();

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
    <ToolbarHeaderWrapper
      title="Add effects"
      description="Apply visual effects to your selected objects and the background. Keeping clicking the same effect for different variations."
      bottomSection={
        <div className="flex mt-1">
          <Button
            color="ghost"
            size="md"
            className={`font-medium bg-black !rounded-full hover:!bg-gradient-to-br ${BLUE_PINK_FILL} border-none`}
            endIcon={<MagicWandFilled size={20} className="text-white " />}
            onClick={handleTogglePreset}>
            Surprise Me
          </Button>
        </div>
      }
      className="pb-4"
    />
  );
}
