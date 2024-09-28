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
import {spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';

const styles = stylex.create({
  container: {
    position: 'absolute',
    width: '100%',
    height: spacing[8],
    pointerEvents: 'none',
  },
});

type CarouselContainerShadowProps = {
  isTop: boolean;
};

const edgeColor = 'rgba(55, 62, 65, 1)';
const transitionColor = 'rgba(55, 62, 65, 0.2)';

export function CarouselContainerShadow({isTop}: CarouselContainerShadowProps) {
  return (
    <div
      {...stylex.props(styles.container)}
      style={{
        background: `linear-gradient(${isTop ? `${edgeColor}, ${transitionColor}` : `${transitionColor}, ${edgeColor}`})`,
        top: isTop ? 0 : undefined,
        bottom: isTop ? undefined : 0,
      }}
    />
  );
}
