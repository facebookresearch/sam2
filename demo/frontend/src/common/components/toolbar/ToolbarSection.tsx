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
  title: string;
  borderBottom?: boolean;
}>;

export default function ToolbarSection({
  children,
  title,
  borderBottom = false,
}: Props) {
  return (
    <div className={`p-6 ${borderBottom && 'border-b border-black'}`}>
      <div className="font-bold ml-2">{title}</div>
      <div className="grid grid-cols-4 gap-2 mt-2 md:mt-6">{children}</div>
    </div>
  );
}
