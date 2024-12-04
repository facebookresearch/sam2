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
import {PropsWithChildren} from 'react';

type Props = PropsWithChildren<{
  className?: string;
  message: string;
  position?: 'left' | 'top' | 'right' | 'bottom';
}>;

/**
 * This is a custom Tooltip component because React Daisy UI does not have an
 * option to *only* show tooltip on large devices.
 */
export default function Tooltip({
  children,
  className = '',
  message,
  position = 'top',
}: Props) {
  return (
    <div
      className={`lg:tooltip tooltip-${position} ${className}`}
      data-tip={message}>
      {children}
    </div>
  );
}
