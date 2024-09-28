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
import {CarouselContainerShadow} from '@/common/components/effects/EffectsCarouselShadow';
import {DemoEffect} from '@/common/components/effects/EffectsUtils';
import useVideoEffect from '@/common/components/video/editor/useVideoEffect';
import type {EffectIndex} from '@/common/components/video/effects/Effects';
import {Effects} from '@/common/components/video/effects/Effects';
import {color, fontSize, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';

type Props = {
  label: string;
  effects: DemoEffect[];
  activeEffect: keyof Effects;
  index: EffectIndex;
};

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[2],
    width: '100%',
  },
  label: {
    fontSize: fontSize['xs'],
    color: '#A6ACB2',
    textAlign: 'center',
  },
  carouselContainer: {
    position: 'relative',
    borderRadius: '8px',
    overflow: 'hidden',
    width: '100%',
    height: '120px',
    backgroundColor: color['gray-700'],
  },
});

export default function EffectsCarousel({
  label,
  effects,
  activeEffect,
  index: effectIndex,
}: Props) {
  const setEffect = useVideoEffect();

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.label)}>{label}</div>
      <div {...stylex.props(styles.carouselContainer)}>
        <CarouselContainerShadow isTop={true} />
        <div className="carousel carousel-vertical w-full h-full text-white">
          <div className={`carousel-item h-6`} />
          {effects.map(({effectName, Icon, title}, index) => {
            const isActive = activeEffect === effectName;
            return (
              <div
                key={index}
                className={`carousel-item flex items-center h-6 gap-2 px-4`}
                onClick={() => setEffect(effectName, effectIndex)}>
                <Icon
                  color={isActive ? '#FB73A5' : undefined}
                  size={18}
                  fontWeight={10}
                />
                <div
                  className={`text-sm ${isActive ? 'text-[#FB73A5] font-bold' : 'font-medium'}`}>
                  {title}
                </div>
              </div>
            );
          })}
          <div className={`carousel-item h-6`} />
        </div>
        <CarouselContainerShadow isTop={false} />
      </div>
    </div>
  );
}
