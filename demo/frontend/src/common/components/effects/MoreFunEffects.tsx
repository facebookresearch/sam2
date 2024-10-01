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
import {moreEffects} from '@/common/components/effects/EffectsUtils';
import EffectVariantBadge from '@/common/components/effects/EffectVariantBadge';
import ToolbarActionIcon from '@/common/components/toolbar/ToolbarActionIcon';
import ToolbarSection from '@/common/components/toolbar/ToolbarSection';
import useVideoEffect from '@/common/components/video/editor/useVideoEffect';
import {EffectIndex} from '@/common/components/video/effects/Effects';
import {activeHighlightEffectAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

export default function MoreFunEffects() {
  const setEffect = useVideoEffect();
  const activeEffect = useAtomValue(activeHighlightEffectAtom);

  return (
    <ToolbarSection title="Selected Objects" borderBottom={true}>
      {moreEffects.map(effect => {
        return (
          <ToolbarActionIcon
            variant="toggle"
            key={effect.title}
            icon={effect.Icon}
            title={effect.title}
            isActive={activeEffect.name === effect.effectName}
            badge={
              activeEffect.name === effect.effectName && (
                <EffectVariantBadge
                  label={`${activeEffect.variant + 1}/${activeEffect.numVariants}`}
                />
              )
            }
            onClick={() => {
              setEffect(effect.effectName, EffectIndex.HIGHLIGHT);
            }}
          />
        );
      })}
    </ToolbarSection>
  );
}
