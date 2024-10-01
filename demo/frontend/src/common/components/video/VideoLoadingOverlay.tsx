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
import {fontSize, fontWeight, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {Loading} from 'react-daisyui';

const styles = stylex.create({
  overlay: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    background: 'rgba(0,0,0,0.5)',
  },
  indicatorContainer: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    display: 'flex',
    alignItems: 'center',
    gap: spacing[4],
    color: 'white',
  },
  indicatorText: {
    color: 'white',
    fontSize: fontSize['sm'],
    fontWeight: fontWeight['medium'],
  },
});

type Props = {
  label?: string;
};

export default function VideoLoadingOverlay({label}: Props) {
  return (
    <div {...stylex.props(styles.overlay)}>
      <div {...stylex.props(styles.indicatorContainer)}>
        <Loading size="sm" />
        <div {...stylex.props(styles.indicatorText)}>
          {label ?? 'Loading video...'}
        </div>
      </div>
    </div>
  );
}
