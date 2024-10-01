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
import useVideo from '@/common/components/video/editor/useVideo';
import {
  activeBackgroundEffectAtom,
  activeHighlightEffectAtom,
} from '@/demo/atoms';
import {useSetAtom} from 'jotai';
import {useCallback, useEffect} from 'react';
import {EffectUpdateEvent} from '../VideoWorkerBridge';
import {EffectOptions} from '../effects/Effect';
import Effects, {EffectIndex, Effects as EffectsType} from '../effects/Effects';

export default function useVideoEffect() {
  const video = useVideo();
  const setBackgroundEffect = useSetAtom(activeBackgroundEffectAtom);
  const setHighlightEffect = useSetAtom(activeHighlightEffectAtom);

  // The useEffect will listen to any effect updates from the worker. The
  // worker is the source of truth, which effect and effect variant is
  // currently applied. The main thread will be notified whenever an effect
  // or effect variant changes.
  useEffect(() => {
    function onEffectUpdate(event: EffectUpdateEvent) {
      if (event.index === EffectIndex.BACKGROUND) {
        setBackgroundEffect(event);
      } else {
        setHighlightEffect(event);
      }
    }
    video?.addEventListener('effectUpdate', onEffectUpdate);
    return () => {
      video?.removeEventListener('effectUpdate', onEffectUpdate);
    };
  }, [video, setBackgroundEffect, setHighlightEffect]);

  return useCallback(
    (name: keyof EffectsType, index: EffectIndex, options?: EffectOptions) => {
      video?.setEffect(name, index, options);
      const effect = Effects[name];
      const effectVariant = options?.variant ?? 0;

      if (index === EffectIndex.BACKGROUND) {
        setBackgroundEffect({
          name,
          variant: effectVariant,
          numVariants: effect.numVariants,
        });
      } else {
        setHighlightEffect({
          name,
          variant: options?.variant ?? 0,
          numVariants: effect.numVariants,
        });
      }
    },
    [video, setBackgroundEffect, setHighlightEffect],
  );
}
