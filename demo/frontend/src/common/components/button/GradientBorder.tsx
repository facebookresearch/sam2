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
import stylex from '@stylexjs/stylex';

import {gradients} from '@/theme/tokens.stylex';

enum GradientTypes {
  fullGradient = 'fullGradient',
  bluePinkGradient = 'bluePinkGradient',
}

type Props = {
  gradientType?: GradientTypes;
  disabled?: boolean;
  rounded?: boolean;
  className?: string;
} & React.DOMAttributes<HTMLDivElement>;

const styles = stylex.create({
  animationHover: {
    ':hover': {
      backgroundPosition: '300% 100%',
    },
  },

  fullGradient: {
    border: '2px solid transparent',
    background: gradients['rainbow'],
    backgroundSize: '100% 400%',
    transition: 'background 0.35s ease-in-out',
  },

  bluePinkGradient: {
    border: '2px solid transparent',
    background: gradients['rainbow'],
  },
});

export default function GradientBorder({
  gradientType = GradientTypes.fullGradient,
  disabled,
  rounded = true,
  className = '',
  children,
}: Props) {
  const gradient = (name: GradientTypes) => {
    if (name === GradientTypes.fullGradient) {
      return styles.fullGradient;
    } else if (name === GradientTypes.bluePinkGradient) {
      return styles.bluePinkGradient;
    }
  };

  return (
    <div
      className={`${stylex(gradient(gradientType), !disabled && styles.animationHover)} ${disabled && 'opacity-30'} ${rounded && 'rounded-full'} ${className}`}>
      {children}
    </div>
  );
}
