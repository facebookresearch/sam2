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
import {right, top} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';

const styles = stylex.create({
  variantBadge: {
    position: 'absolute',
    top: top[1],
    right: right[1],
    backgroundColor: '#280578',
    color: '#D2D2FF',
    fontVariantNumeric: 'tabular-nums',
    paddingHorizontal: 4,
    paddingVertical: 1,
    fontSize: 9,
    borderRadius: 6,
    fontWeight: 'bold',
  },
});

type Props = {
  label: string;
};

export default function VariantBadge({label}: Props) {
  return <div {...stylex.props(styles.variantBadge)}>{label}</div>;
}
