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
import {ReactNode} from 'react';
import ToolbarProgressChip from './ToolbarProgressChip';

type Props = {
  title: string;
  description?: string;
  bottomSection?: ReactNode;
  showProgressChip?: boolean;
  className?: string;
};

export default function ToolbarHeaderWrapper({
  title,
  description,
  bottomSection,
  showProgressChip = true,
  className,
}: Props) {
  return (
    <div
      className={`flex flex-col gap-2 p-8 border-b border-b-black ${className}`}>
      <div className="flex items-center">
        {showProgressChip && <ToolbarProgressChip />}
        <h2 className="text-xl">{title}</h2>
      </div>

      {description != null && (
        <div className="flex-1 text-gray-400">{description}</div>
      )}
      {bottomSection != null && bottomSection}
    </div>
  );
}
